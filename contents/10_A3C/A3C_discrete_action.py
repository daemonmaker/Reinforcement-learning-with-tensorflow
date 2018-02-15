"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0

TODO
- Add forward predictions
- Add cos activation
- Add evaluation at every X epochs
- Replace logger
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt

from gym.wrappers.time_limit import TimeLimit
import argparse
#from baselines import logger
from skimage.color import rgb2grey
from skimage.transform import resize
import copy
import time
import numpy as np
from keras import metrics
import keras.backend as KK
from keras.layers import UpSampling2D


img_lock = threading.Lock()
log_lock = threading.Lock()

GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_EP_STEPS = 1000
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 10 #100 #10
GAMMA = 0.9 #0.99 # 0.9
ENTROPY_BETA = 0.001 # 0.001
GLOBAL_RUNNING_R = []
GLOBAL_R = []
GLOBAL_EP = 0

N_S = 0
IMAGE_SHAPE = None
N_A = 0
N_VAE = 6
N_FORWARD = 3
A_BOUND = []
CONTINUOUS = False
TASKS = []
OPT = {}
FORWARD_PREDICT = 'vae'
RECONSTRUCT = 'vae'


def get_img(env, fn, *args):
    img_lock.acquire()
    results = fn(env, *args)
    if CONTINUOUS and type(env.env) is gym.envs.classic_control.pendulum.PendulumEnv:
        img = env.render(mode='rgb_array_no_arrow')
    else:
        img = env.render(mode='rgb_array')
    img_lock.release()
    img = resize(img, IMAGE_SHAPE)
    return img, results

def env_reset_obs(env):
    s = env.reset()
    #if type(env.env) is gym.envs.classic_control.pendulum.PendulumEnv:
    #    env.env.state = np.array([np.pi, 0])
    #    s = env.env._get_obs()
    return s, None

def env_reset_img(env):
    img, results = get_img(env, env_reset_obs)
    return img, results

def env_get_obs(env):
    return env.env._get_obs()

def env_get_img(env):
    img, _ = get_img(env, env_get_obs)
    return img

def env_step_obs(env, a):
    return env.step(a)

def env_step_img(env, a):
    img, results = get_img(env, env_step_obs, a)
    return img, results[1], results[2], results[3]

env_reset_fn = env_reset_obs
env_get_obs_fn = env_get_obs
env_step_fn = env_step_obs

def summarize_weights():
    temp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(len(temp))
    for layer in temp:
        print(layer)

class ACNet(object):
    def __init__(self, scope, w_init, soft_sharing_coeff, gradient_clip, globalAC=None, hard_share=None, stack=1, forward_predict=False, reconstruct=False, batch_normalize=False, obs_diff=False):
        self.scope = scope
        self.w_init = w_init
        self.gradient_clip = gradient_clip
        self.hard_share = hard_share
        self.stack = stack
        self.batch_normalize = batch_normalize
        self.obs_diff = obs_diff

        def input_placeholders():
            if IMAGE_SHAPE is not None:
                s = tuple([tf.placeholder(tf.float32, [None, ] + list(IMAGE_SHAPE), 'S') for _ in range(self.stack)])
            else:
                s = tuple([tf.placeholder(tf.float32, [None, N_S], 'S') for _ in range(self.stack)])
            return s

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = input_placeholders()

                self.outputs, self.params = self._build_net(scope)

                if 'actor' in TASKS and CONTINUOUS:
                    mu = self.outputs['actor'][0]
                    sigma = self.outputs['actor'][1]

                    with tf.name_scope('wrap_a_out'):
                        mu, sigma = mu*A_BOUND[1], sigma + 1e-4

                    normal_dist = tf.distributions.Normal(mu, sigma)

                    with tf.name_scope('choose_a'):  # use local params to choose action
                        self.A = mu # TODO DWEBB Should we use the mean or sample? Using this works better because it doesn't inject noise.
                        #self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])

        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = input_placeholders()
                forward_shape = [None, N_S] if IMAGE_SHAPE is None else [None, ] + list(IMAGE_SHAPE)
                self.f_target = tf.placeholder(tf.float32, forward_shape, 'forward_prediction_targets')
                action_shape = [None, N_A] if CONTINUOUS else [None, ]
                a_dtype = tf.float32 if CONTINUOUS else tf.int32
                self.a_his = tf.placeholder(a_dtype, action_shape, 'actions')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'value_targets')

                self.outputs, self.params = self._build_net(scope)
                self.losses = {}

                if 'actor' in TASKS and CONTINUOUS:
                    mu = self.outputs['actor'][0]
                    sigma = self.outputs['actor'][1]

                if 'critic' in TASKS:
                    td = tf.subtract(self.v_target, self.outputs['critic'], name='TD_error')
                    self.t_td = td
                    with tf.name_scope('c_loss'):
                        self.losses['critic'] = tf.reduce_mean(tf.square(td))
                        if soft_sharing_coeff['critic'] > 0:
                            self.losses['critic'] += soft_sharing_coeff['critic']*tf.nn.l2_loss(self.l_a - self.l_c)

                if 'actor' in TASKS:
                    if CONTINUOUS:
                        with tf.name_scope('wrap_a_out'):
                            mu, sigma = mu*A_BOUND[1], sigma + 1e-4
    
                        normal_dist = tf.distributions.Normal(mu, sigma)
    
                    with tf.name_scope('a_loss'):
                        if CONTINUOUS:
                            log_prob = normal_dist.log_prob(self.a_his)
                        else:
                            log_prob = tf.reduce_sum(tf.log(self.outputs['actor']) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)

                        if 'critic' in TASKS:
                            exp_v = log_prob * tf.stop_gradient(td)
                        else:
                            exp_v = log_prob # TODOD DWEBB is this the correct way to remove the ritic?

                        entropy_beta = ENTROPY_BETA
                        if CONTINUOUS:
                            entropy = normal_dist.entropy()  # encourage exploration
                            #entropy_beta = np.abs(np.random.randn(1))*ENTROPY_BETA
                        else:
                            entropy = -tf.reduce_sum(self.outputs['actor'] * tf.log(self.outputs['actor'] + 1e-5),
                                                     axis=1, keep_dims=True)  # encourage exploration
                        print('entropy_beta: ', entropy_beta)
                        self.t_log_prob = log_prob
                        self.t_entropy = entropy
                        self.t_exp_v = exp_v
                        self.exp_v = entropy_beta * entropy + exp_v
                        self.t_exp_v2 = self.exp_v
                        self.losses['actor'] = tf.reduce_mean(-self.exp_v)
                        if soft_sharing_coeff['actor'] > 0:
                            self.losses['actor'] += soft_sharing_coeff['actor']*tf.nn.l2_loss(self.l_a - self.l_c)
    
                    if CONTINUOUS:
                        with tf.name_scope('choose_a'):  # use local params to choose action
                            self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])
    
                if 'forward_predict' in TASKS:
                    with tf.name_scope('forward_prediction_loss'):
                        if FORWARD_PREDICT == 'vae':
                            raise NotImplementedError()
                        elif FORWARD_PREDICT == 'mse':
                            self.losses['forward_predict'] = tf.reduce_mean(tf.losses.mean_squared_error(self.f_target, self.outputs['forward_predict']))
                        else:
                            raise ValueError('Unknown forward prediction type.')

                if 'reconstruct' in TASKS:
                    with tf.name_scope('reconstruction_loss'):
                        if RECONSTRUCT == 'mse':
                            self.losses['reconstruct'] = tf.reduce_mean(tf.losses.mean_squared_error(self.s[-1], self.outputs['reconstruct']))
                        elif RECONSTRUCT == 'bce':
                            self.losses['reconstruct'] = tf.reduce_mean(tf.keras.backend.binary_crossentropy(self.s[-1], self.outputs['reconstruct']))
                        elif RECONSTRUCT == 'vae':
                            x = KK.flatten(self.s[0])
                            x_decoded_mean_squash_flat = KK.flatten(self.outputs['reconstruct'])

                            if IMAGE_SHAPE is not None:
                                img_rows = IMAGE_SHAPE[0]
                                img_cols = IMAGE_SHAPE[0]
                                shape_coeff = img_rows*img_cols
                            else:
                                shape_coeff = N_S
                            xent_loss = shape_coeff * metrics.binary_crossentropy(x, x_decoded_mean_squash_flat)
                            kl_loss = - 0.5 * KK.mean(1 + self.vae_z_log_var - KK.square(self.vae_z_mean) - KK.exp(self.vae_z_log_var), axis=-1)

                            self.losses['reconstruct'] = 10*KK.mean(xent_loss + kl_loss)
                        else:
                            raise ValueError('Unknown reconstruction loss')
                        if soft_sharing_coeff['reconstruct'] > 0:
                            self.losses['reconstruct'] += soft_sharing_coeff['reconstruct']*tf.nn.l2_loss(self.l_a - self.l_r)
                            self.losses['reconstruct'] += soft_sharing_coeff['reconstruct']*tf.nn.l2_loss(self.l_c - self.l_r)

                self.grads = {}
                with tf.name_scope('local_grad'):
                    for task in TASKS:
                        if task in TASKS:
                            self.grads[task] = tf.gradients(self.losses[task], self.params[task])
                            if gradient_clip[task] > 0:
                                self.grads[task], _ = tf.clip_by_global_norm(self.grads[task], self.gradient_clip[task])

                self.pull_params_op = {}
                self.update_op = {}
                with tf.name_scope('sync'):
                    with tf.name_scope('pull'):
                        for task in TASKS:
                            self.pull_params_op[task] = [l_p.assign(g_p) for l_p, g_p in zip(self.params[task], globalAC.params[task])]
                    with tf.name_scope('push'):
                        for task in TASKS:
                            self.update_op[task] = OPT[task].apply_gradients(zip(self.grads[task], globalAC.params[task]))

    def _batch_normalize(self, inputs):
        if self.batch_normalize:
            return tf.layers.batch_normalization(inputs)
        else:
            return inputs

    def _build_obs_processor(self, obs, n_out, reuse, scope):
        with tf.variable_scope(scope):
            if IMAGE_SHAPE is not None:
                conv1 = tf.layers.conv2d(obs, 32, 5, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c1')
                pool1 = tf.layers.max_pooling2d(conv1, 2, 1, name='p1')
                conv2 = tf.layers.conv2d(pool1, 64, 5, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c2')
                pool2 = tf.layers.max_pooling2d(conv2, 2, 1, name='p2')
                conv3 = tf.layers.conv2d(pool2, 64, 5, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c3')
                pool3 = tf.layers.max_pooling2d(conv3, 3, 1, name='p3')
                flattened_conv = tf.layers.flatten(conv2, name='flattened_conv')
                obs = tf.layers.dense(flattened_conv, 360, tf.nn.relu6, kernel_initializer=self.w_init, reuse=reuse, name='fc')
            final = tf.layers.dense(obs, n_out, tf.nn.relu6, kernel_initializer=self.w_init, reuse=reuse, name='processed_obs')
        return final

    def _process_inputs(self, obses, n_out, reuse=False, share_processor=False):
        if share_processor:
            scope_template = 'input_processor'
        else:
            scope_template = 'input_processor_{}'
        processed_inputs = []
        for idx, obs in enumerate(obses):
            processed_inputs.append(self._build_obs_processor(obs, n_out, reuse, scope=scope_template.format(idx)))
            if share_processor: reuse=True
        processed_inputs = tf.concat(processed_inputs, -1)
        return processed_inputs

    def _build_conv(self, inputs, w_init, reuse=False):
        with tf.variable_scope('convs'):
            if self.obs_diff and self.stack == 2:
                diff = inputs[1] - inputs[0]
                concat_inputs = tf.concat([inputs[1], diff], -1)
            else:
                concat_inputs = tf.concat(inputs, -1)

            conv1 = tf.layers.conv2d(self._batch_normalize(concat_inputs), 16, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c1')
            pool1 = tf.layers.max_pooling2d(conv1, 2, 1, reuse=reuse, name='p1')
            conv2 = tf.layers.conv2d(self._batch_normalize(pool1), 16, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c2')
            pool2 = tf.layers.max_pooling2d(conv2, 2, 1, reuse=reuse, name='p2')
            conv3 = tf.layers.conv2d(self._batch_normalize(pool2), 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c3')
            conv4 = tf.layers.conv2d(conv3, 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c4')
            pool4 = tf.layers.max_pooling2d(conv4, 2, 1, reuse=reuse, name='p3')
            conv5 = tf.layers.conv2d(self._batch_normalize(pool4), 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c5')
            conv6 = tf.layers.conv2d(conv5, 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c6')
            self.flattened_conv = tf.layers.flatten(conv6, name='flattened_conv')
            fc = tf.layers.dense(self.flattened_conv, 512, tf.nn.relu6, kernel_initializer=w_init, reuse=reuse, name='fc')
            inputs = tf.layers.dense(fc, 512, tf.nn.relu6, kernel_initializer=w_init, reuse=reuse, name='inputs')
            '''
            # TODO DWEBB add stack support here
            diff = inputs[1] - inputs[0]
            inputs = tf.concat([inputs[1], diff], -1)
            conv1 = tf.layers.conv2d(inputs, 64, 8, name='c1')
            #pool1 = tf.layers.max_pooling2d(conv1, 5, 1, name='p1')
            relu1 = tf.nn.relu(conv1, name='relu1')
            conv2 = tf.layers.conv2d(relu1, 32, 8, name='c2')
            pool2 = tf.layers.max_pooling2d(conv2, 5, 1, name='p2')
            relu2 = tf.nn.relu(pool2, name='relu2')
            conv3 = tf.layers.conv2d(relu2, 32, 5, name='c3')
            pool3 = tf.layers.max_pooling2d(conv3, 5, 1, name='p3')
            relu3 = tf.nn.relu(pool3, name='relu3')
            conv4 = tf.layers.conv2d(relu3, 32, 5, name='c4')
            pool4 = tf.layers.max_pooling2d(conv4, 5, 1, name='p4')
            relu4 = tf.nn.relu(pool4, name='relu4')
            inputs = tf.layers.flatten(relu4, name='p3')
            '''
        return inputs

    def _build_deconv(self, inputs, reuse=False):
        with tf.variable_scope('deconvs'):
            # TODO DWEBB fix these magic numbers!
            temp = tf.layers.dense(inputs, 25, reuse=reuse, name='temp')
            inputs = tf.reshape(temp, [-1, 5, 5, 1], name='reshaped_flat')
            deconv1 = tf.layers.conv2d_transpose(inputs, 32, 5, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d1')
            deconv2 = tf.layers.conv2d_transpose(deconv1, 64, 5, strides=(2,2), padding='same', activation=tf.nn.relu, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d2')
            inputs = tf.layers.conv2d_transpose(deconv2, 3, 8, strides=(2,2), padding='same', activation=tf.sigmoid, kernel_initializer=self.w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d3')
        return inputs

    def _vae_sample(self, h):
        z_mean = tf.layers.dense(h, N_VAE, tf.nn.tanh, kernel_initializer=self.w_init, name='mu')
        z_log_var = tf.layers.dense(h, N_VAE, tf.nn.softplus, kernel_initializer=self.w_init, name='sigma')
        z = tf.squeeze(z_mean + tf.exp(z_log_var)*tf.distributions.Normal(tf.zeros(tf.shape(z_mean)), tf.ones(tf.shape(z_mean))).sample(1), 0)
        return z_mean, z_log_var, z

    def _build_hard_share(self, scope, n_out = 300, share_input_processor=False):
        outputs = {}
        params = {}

        with tf.variable_scope('input_processor'):
            l_p = self._process_inputs(self.s, self.w_init, n_out, share_processor=share_input_processor)
        input_processor_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/input_processor')

        if 'actor' in TASKS:
            with tf.variable_scope('actor'):
                if CONTINUOUS:
                    mu = tf.layers.dense(l_p, N_A, tf.nn.tanh, kernel_initializer=self.w_init, name='mu')
                    sigma = tf.layers.dense(l_p, N_A, tf.nn.softplus, kernel_initializer=self.w_init, name='sigma')
                    a_prob = (mu, sigma)
                else:
                    a_prob = tf.layers.dense(l_p, N_A, tf.nn.softmax, kernel_initializer=self.w_init, name='ap')
            outputs['actor'] = a_prob
            params['actor'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor') + input_processor_params

        if 'critic' in TASKS:
            with tf.variable_scope('critic'):
                v = tf.layers.dense(l_p, 1, kernel_initializer=self.w_init, name='v')  # state value
            outputs['critic'] = v
            params['critic'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic') + input_processor_params

        if 'forward_predict' in TASKS:
            with tf.variable_scope('forward_predict'):
                fp_h = l_p
                fp_h = tf.layers.dense(fp_h, n_out, tf.nn.relu, kernel_initializer=self.w_init, name='fp')
                if FORWARD_PREDICT == 'vae':
                    self.fp_z_mean, self.fp_z_log_var, self.fp_z = self._vae_sample(fp_h)
                    fp_h = self.fp_z
                if IMAGE_SHAPE is not None:
                    forward_prediction = self._build_deconv(fp_h, reuse=False) # TODO DWEBB Address sharing of deconv parameters...
                else:
                    forward_prediction = tf.layers.dense(fp_h, N_S, kernel_initializer=self.w_init,  name='r')
            outputs['forward_prediction'] = forward_prediction
            params['forward_predict'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/forward_predict') + input_processor_params

        if 'reconstruct' in TASKS:
            with tf.variable_scope('reconstruct'):
                r_p = l_p
                if RECONSTRUCT == 'vae':
                    self.vae_z_mean, self.vae_z_log_var, self.vae_z = self._vae_sample(r_p)
                    r_p = self.vae_z
                if IMAGE_SHAPE is not None:
                    reconstruct = self._build_deconv(r_p, reuse=False) # TODO DWEBB Address sharing of deconv parameters...
                else:
                    reconstruct = tf.layers.dense(r_p, N_S, kernel_initializer=w_init,  name='r')
            outputs['reconstruct'] = reconstruct
            params['reconstruct'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct') + input_processor_params
    
        return outputs, params

    def _build_soft_share(self, scope, n_hiddens, share_input_processor=False):
        # TODO DWEBB Store links to the different layers so they can be penalized with soft sharing schemes, e.g. via the regularization in the loss.
        outputs = {}
        params = {}
        if 'actor' in TASKS:
            with tf.variable_scope('actor'):
                self.l_a = self._process_inputs(self.s, n_hiddens['actor'], share_processor=share_input_processor)
                if CONTINUOUS:
                    mu = tf.layers.dense(self.l_a, N_A, tf.nn.tanh, kernel_initializer=self.w_init, name='a_mu')
                    sigma = tf.layers.dense(self.l_a, N_A, tf.nn.softplus, kernel_initializer=self.w_init, name='a_sigma')
                    a_prob = (mu, sigma)
                else:
                    a_prob = tf.layers.dense(self.l_a, N_A, tf.nn.softmax, kernel_initializer=self.w_init, name='ap')
                outputs['actor'] = a_prob
                params['actor'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')

        if 'critic' in TASKS:
            with tf.variable_scope('critic'):
                self.l_c = self._process_inputs(self.s, n_hiddens['critic'], share_processor=share_input_processor)
                v = tf.layers.dense(self.l_c, 1, kernel_initializer=self.w_init, name='v')  # state value
                outputs['critic'] = v
                params['critic'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

        if 'forward_predict' in TASKS:
            with tf.variable_scope('forward_predict'):
                self.l_f = self._process_inputs(self.s, n_hiddens['foward_predict'], share_processor=share_input_processor)
                fp_h = tf.layers.dense(self.l_f, n_hiddens['f'], tf.nn.relu, kernel_initializer=w_init, name='fp_h')
                if FORWARD_PREDICT == 'vae':
                    self.fp_z_mean, self.fp_z_log_var, self.fp_z = self._vae_sample(fp_h)
                    fp_h = self.vae_z
                if IMAGE_SHAPE is not None:
                    forward_prediction = self._build_deconv(fp_h, reuse=False) # TODO DWEBB Address sharing of deconv parameters...
                else:
                    forward_prediction = tf.layers.dense(fp_h, N_S, kernel_initializer=w_init,  name='fp')
                outputs['forward_predict'] = forward_prediction
                params['forward_predict'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/forward_predict')

        if 'reconstruct' in TASKS:
            with tf.variable_scope('reconstruct'):
                self.l_r = self._process_inputs(self.s, n_hiddens['reconstruct'], share_processor=share_input_processor)
                if RECONSTRUCT == 'vae':
                    self.vae_z_mean, self.vae_z_log_var, self.vae_z = self._vae_sample(self.l_r)
                    self.l_r = self.vae_z
                if IMAGE_SHAPE is not None:
                    reconstruct = self._build_deconv(self.l_r, reuse=False)
                else:
                    reconstruct = tf.layers.dense(self.l_r, N_S, kernel_initializer=self.w_init, name='r')
                outputs['reconstruct'] = reconstruct
                params['reconstruct'] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct')

        return outputs, params

    def _build_net(self, scope):
        n_hiddens = {'actor': 200, 'critic': 100, 'reconstruct': 6, 'forward_predict': 100}
        if self.hard_share is not None:
            n_out = np.sum([n_hiddens[task] for task in TASKS])
            outputs, params = self._build_hard_share(scope, n_out=n_out, share_input_processor=True)
        else:
            outputs, params = self._build_soft_share(scope, n_hiddens=n_hiddens, share_input_processor=False)

        return outputs, params

    def _build_net_old(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        inputs = self.s
        s_params = []
        reconstruct = ''
        if self.hard_share is not None:
            if IMAGE_SHAPE is not None:
                inputs = self._build_conv(inputs, w_init)
            else:
                inputs = tf.concat(inputs, 1)

            if self.hard_share == 'equal_params':
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 71, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    if CONTINUOUS:
                        mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                        sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                        a_prob = (mu, sigma)
                    else:
                        a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                if RECONSTRUCT:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if IMAGE_SHAPE is not None:
                            reconstruct = self._build_deconv(l_r, w_init)
                        else:
                            reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')  # state value
            elif False:
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    if CONTINUOUS:
                        mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                        sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                        a_prob = (mu, sigma)
                    else:
                        a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                if RECONSTRUCT:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if IMAGE_SHAPE is not None:
                            reconstruct = self._build_deconv(l_r, w_init)
                        else:
                            reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')  # state value
            else:
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 200, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    if CONTINUOUS:
                        mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                        sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                        a_prob = (mu, sigma)
                    else:
                        a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
                if RECONSTRUCT:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if IMAGE_SHAPE is not None:
                            reconstruct = self._build_deconv(l_r, w_init)
                        else:
                            reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')  # state value

            s_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/shared')
        else:
            if IMAGE_SHAPE is not None:
                concat_inputs = self._build_conv(inputs, w_init)
            else:
                concat_inputs = tf.concat(inputs, 1)

            with tf.variable_scope('actor'):
                '''
                if IMAGE_SHAPE is not None:
                    concat_inputs = self._build_conv(inputs, w_init)
                else:
                    concat_inputs = tf.concat(inputs, 1)
                '''
                l_a = tf.layers.dense(concat_inputs, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
                #l_a = tf.layers.dense(l_a, 100, tf.nn.relu6, kernel_initializer=w_init, name='la1')
                self.l_a = l_a
                if CONTINUOUS:
                    mu = tf.layers.dense(l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                    sigma = tf.layers.dense(l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                    a_prob = (mu, sigma)
                else:
                    a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            with tf.variable_scope('critic'):
                '''
                if IMAGE_SHAPE is not None:
                    concat_inputs = self._build_conv(inputs, w_init)
                else:
                    concat_inputs = tf.concat(inputs, 1)
                '''
                l_c = tf.layers.dense(concat_inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                #l_c = tf.layers.dense(l_c, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc1')
                self.l_c = l_c
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            if RECONSTRUCT:
                with tf.variable_scope('reconstruct'):
                    '''
                    if IMAGE_SHAPE is not None:
                        concat_inputs = self._build_conv(inputs, w_init)
                    else:
                        concat_inputs = tf.concat(inputs, 1)
                    '''
                    l_r = tf.layers.dense(concat_inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                    l_r = tf.layers.dense(l_r, 100, tf.nn.relu6, kernel_initializer=w_init, name='lr1')
                    self.l_r = l_r
                    if IMAGE_SHAPE is not None:
                        reconstruct = self._build_deconv(l_r, w_init)
                    else:
                        reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        if IMAGE_SHAPE is not None:
            i_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/convs')
            if RECONSTRUCT:
                i_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/deconvs')
        else:
            i_params = []
        if RECONSTRUCT:
            r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct')
        else:
            r_params = []
        return a_prob, v, reconstruct, a_params + s_params + i_params, c_params + s_params + i_params, r_params + s_params + i_params

    def get_stats(self, feed_dict):
        return SESS.run([self.a_loss, self.c_loss, self.t_td, self.c_loss, self.t_log_prob, self.t_exp_v, self.t_entropy, self.t_exp_v2, self.a_loss, self.a_grads, self.c_grads], feed_dict)

    def _build_obs_columns(self, buffer_s):
        obs_columns = [buffer_s[idx:-(self.stack-(idx+1))] for idx in range(self.stack-1)]
        obs_columns.append(buffer_s[self.stack-1:])

        feed_dict = {var: obs for var, obs in zip(self.s, obs_columns)}

        return feed_dict, (0,)

    def update_global(self, feed_dict):  # run by a local
        results_map = {}
        stats = []
        if hasattr(self, 't_exp_v'):
            results_map['exp_v'] = 0
            stats.append(self.t_exp_v)
        offset = len(stats)
        update_ops = []
        for idx, task in enumerate(TASKS):
            stats.append(self.losses[task])
            update_ops.append(self.update_op[task])
            results_map[task] = idx + offset
        results = SESS.run(stats + update_ops, feed_dict)[:len(stats)]
        output_stats = {}
        for key, idx in results_map.items():
            output_stats[key] = results[idx]

        if 'exp_v' in output_stats:
            output_stats['exp_v'] = output_stats['exp_v'][0, 0]
        return output_stats

    def pull_global(self):  # run by a local
        SESS.run(tuple(self.pull_params_op.values()))

    def _choose_action(self, s):  # run by a local
        feed_dict, _ = self._build_obs_columns(s)
        if CONTINUOUS:
            action = np.squeeze(SESS.run(self.A, feed_dict=feed_dict)[0])
        else:
            prob_weights = SESS.run(self.outputs['actor'], feed_dict=feed_dict)
            action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def choose_action(self, s):  # run by a local
        if 'actor' in TASKS:
            action = self._choose_action(s)
        elif CONTINUOUS:
            action = ((A_BOUND[1] - A_BOUND[0])*np.random.rand(N_A) + A_BOUND[0])[0]
        else:
            action = np.random.randint(N_A)
        return action

    def reconstruct(self, buffer_s):  # run by a local
        feed_dict, buffer_s_ = self._build_obs_columns(buffer_s)
        return SESS.run(self.outputs['reconstruct'], feed_dict=feed_dict), buffer_s_

class Worker(object):
    def __init__(self, name, w_init, globalAC, soft_sharing_coeff, gradient_clip, hard_share=None, debug=False, stack=1, hold=1, batch_normalize=False, obs_diff=False):
        self.env = gym.make(GAME).unwrapped
        self.env = TimeLimit(self.env, max_episode_steps=MAX_EP_STEPS)
        self.name = name
        self.AC = ACNet(name, w_init, soft_sharing_coeff, gradient_clip, globalAC=globalAC, hard_share=hard_share, stack=stack,  batch_normalize=batch_normalize, obs_diff=obs_diff)
        self.debug = debug
        self.stack = stack
        self.hold = hold

        if IMAGE_SHAPE is not None:
            self.buffer_s = np.zeros((UPDATE_GLOBAL_ITER + self.stack - 1, ) + IMAGE_SHAPE)
        else:
            self.buffer_s = np.zeros((UPDATE_GLOBAL_ITER + self.stack - 1, N_S))
        self.buffer_a = np.zeros((UPDATE_GLOBAL_ITER, N_A))
        self.buffer_r = np.zeros((UPDATE_GLOBAL_ITER, 1))
        self.buffer_v_target = np.zeros((UPDATE_GLOBAL_ITER, 1))

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_R, GLOBAL_EP, MAX_GLOBAL_EP
        global env_reset_fn, env_get_obs_fn, env_step_fn

        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s, _ = env_reset_fn(self.env)

            buffer_s = [s]*(self.stack-1)
            buffer_idx = 0
            for s_idx in range(self.stack-1):
                self.buffer_s[s_idx] = s
            ep_r = 0
            action_count = 0
            for ep_t in range(MAX_EP_STEPS):
                self.buffer_s[buffer_idx+self.stack-1] = s
                if action_count % self.hold == 0:
                    a = self.AC.choose_action(self.buffer_s[buffer_idx:buffer_idx+self.stack])
                #if self.name == 'W_0':
                #    print('ep_r: ', ep_r, '\taction: ', a)
                action_count += 1
                s_, r, done, info = env_step_fn(self.env, np.array([a])) # HACK

                if CONTINUOUS:
                    done = True if ep_t == MAX_EP_STEPS - 1 else False
                elif done: r = -2000 #-5

                ep_r += r
                self.buffer_a[buffer_idx] = a
                if CONTINUOUS:
                    buffer_r.append((r+8)/8)
                    self.buffer_r[buffer_idx] = (r+8)/8
                else:
                    buffer_r.append(r)
                    self.buffer_r[buffer_idx] = r

                if total_step % UPDATE_GLOBAL_ITER == 0 or done: # or np.random.rand() > 0.99:   # update global and assign to local net
                    #print("Updating {} {}...".format(ep_t, total_step))
                    feed_dict, _ = self.AC._build_obs_columns(self.buffer_s[:buffer_idx+self.stack])

                    if 'critic' in TASKS:
                        if done:
                            v_s_ = 0   # terminal
                        else:
                            obs_hist = self.buffer_s[buffer_idx:buffer_idx+self.stack]
                            obs_feed_dict = {var: obs[np.newaxis, :] for var, obs in zip(self.AC.s, obs_hist)}
                            v_s_ = SESS.run(self.AC.outputs['critic'], feed_dict=obs_feed_dict)[0, 0]

                        for idx, r in enumerate(self.buffer_r[buffer_idx::-1]):    # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            self.buffer_v_target[idx] = v_s_

                        feed_dict[self.AC.v_target] = np.flip(self.buffer_v_target[:buffer_idx+1], 0)

                    if 'actor' in TASKS:
                        feed_dict[self.AC.a_his] = self.buffer_a[:buffer_idx+1]

                    '''
                    print(len(buffer_s))
                    print(len(buffer_a))
                    print(len(buffer_v_target))
                    '''
                    '''
                    for idx in range(self.stack):
                        print(obs_columns[idx].shape)
                        print(obs_columns[idx])
                    print(np.vstack(buffer_s_).shape)
                    print(np.vstack(buffer_s_))
                    print(len(buffer_a))

                    import ipdb; ipdb.set_trace()
                    '''

                    if self.debug and self.name == 'W_0':
                        a_loss, c_loss, t_td, c_loss, t_log_prob, t_exp_v, t_entropy, t_exp_v2, a_loss, a_grads, c_grads = self.AC.get_stats(feed_dict)
                        #print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss.shape, " ", c_loss, "\ttd: ", t_td.shape, " ", t_td, "\tlog_prob: ", t_log_prob.shape, " ", t_log_prob, "\texp_v: ", t_exp_v.shape, " ", t_exp_v, "\tentropy: ", t_entropy.shape, " ", t_entropy, "\texp_v2: ", t_exp_v2.shape, " ", t_exp_v2, "\ta_grads: ", [np.sum(weights) for weights in a_grads], "\tc_grads: ", [np.sum(weights) for weights in c_grads])
                        print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss)
                    update_outputs = self.AC.update_global(feed_dict)

                    buffer_s = buffer_s[-(self.stack-1):] if self.stack > 1 else []
                    buffer_a, buffer_r = [], []
                    buffer_idx = -1
                    self.AC.pull_global()

                s = s_
                total_step += 1
                buffer_idx += 1
                if done:
                    GLOBAL_R.append(ep_r)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        if CONTINUOUS:
                            GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                        else:
                            GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)

                    print(self.name, '\tep: ', GLOBAL_EP, '\tep_r_av: ', GLOBAL_RUNNING_R[-1], '\tep_r: ', ep_r, end='')
                    for name, update_output in update_outputs.items():
                        print('\t', name, ': ', update_output, end='')
                    print('')
                    '''
                    log_lock.acquire()
                    logger.record_tabular("global_ep", GLOBAL_EP)
                    logger.record_tabular("name", self.name)
                    logger.record_tabular("ep_r", ep_r)
                    logger.record_tabular("ep_r_weighted", GLOBAL_RUNNING_R[-1])
                    logger.record_tabular("c_loss", c_loss)
                    logger.record_tabular("a_loss", a_loss)
                    if RECONSTRUCT:
                        logger.record_tabular("r_loss", r_loss)
                    logger.record_tabular("entropy", entropy)
                    logger.dump_tabular()
                    log_lock.release()
                    '''

                    GLOBAL_EP += 1
                    break


def parse_args():
    global TASKS, GAME, A_BOUND, ENTROPY_BETA, MAX_EP_STEPS, UPDATE_GLOBAL_ITER, MAX_GLOBAL_EP, N_S, N_A, IMAGE_SHAPE, CONTINUOUS, N_WORKERS, FORWARD_PREDICT
    global env_reset_fn, env_get_obs_fn, env_step_fn

    initializers = ['orthogonal', 'standard_normal']

    parser = argparse.ArgumentParser(description='Run A3C on discrete cart-pole.')
    parser.add_argument('--game', default='CartPole-v0', help='Which environment to learn to control.')
    parser.add_argument('--entropy_beta', type=float, default=0.01, help="Value for the entropy beta term.")
    parser.add_argument('--hard_share', type=str, default='none', help="Indicates whether the models should have an equal number of parameters ('equal_params'), an equal number of hidden units ('equal_hiddens'), or no sharing ('none' -- default).")
    parser.add_argument('--soft_share', type=float, default=0.0, help='Enables soft sharing of both actor and critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_actor', type=float, default=0.0, help='Enables soft sharing of actor parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_critic', type=float, default=0.0, help='Enables soft sharing of critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_forward_predict', type=float, default=0.0, help='Enables soft sharing of forward prediction parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_reconstruct', type=float, default=0.0, help='Enables soft sharing of reconstruction parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--gradient_clip', type=float, default=0.0, help='Enables gradient clipping of actor and critic parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_actor', type=float, default=0.0, help='Enables gradient clipping of actor parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_critic', type=float, default=0.0, help='Enables gradient clipping of critic parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_forward', type=float, default=0.0, help='Enables gradient clipping of forward prediction parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_reconstruct', type=float, default=0.0, help='Enables gradient clipping of reconstruction parameters with the specificied maximum gradient.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enables debugging output.')
    parser.add_argument('--optimizer', default='adagrad', help='Which optimizer to use: rmsprop, adam, adagrad.')
    parser.add_argument('--lr', type=float, default=0.0, help='Sets the learning rate of the actor and critic.')
    parser.add_argument('--lr_a', type=float, default=0.001, help='Sets the learning rate of the actor.')
    parser.add_argument('--lr_c', type=float, default=0.001, help='Sets the learning rate of the critic.')
    parser.add_argument('--lr_f', type=float, default=0.001, help='Sets the learning rate of the forward prediction.')
    parser.add_argument('--lr_r', type=float, default=0.001, help='Sets the learning rate of the reconstruction.')
    parser.add_argument('--log', default=False, action='store_true', help='Enables logging.')
    parser.add_argument('--max_global_ep', type=int, default=500, help='Sets the maximum number of episodes to be executed across all threads.')
    parser.add_argument('--update_global_iter', type=int, default=100, help='How frequently to update the global AC.')
    parser.add_argument('--max_ep_steps', type=int, default=1000, help='The number of time steps per episode before calling the episode done.')
    parser.add_argument('--image_shape', nargs='*', default=None, help='Designates that images shoud be used in lieu of observations and what shpae to use for them.')
    parser.add_argument('--debug_worker', default=False, action='store_true')
    parser.add_argument('--disable_critic', default=False, action='store_true')
    parser.add_argument('--disable_actor', default=False, action='store_true')
    parser.add_argument('--reconstruct', default=False, help='Enables observation reconstruction as an additional learning signal. Options are \'mse\', \'bce\' and \'vae\'.')
    parser.add_argument('--batch_normalize', default=False, action='store_true', help='Enables batch normalization of the convolutional layers.')
    parser.add_argument('--stack', type=int, default=1, help='Number of observations to use for state.')
    parser.add_argument('--hold', type=int, default=1, help='Number of time steps to hold the control.')
    parser.add_argument('--obs_diff', default=False, action='store_true', help='Requires stack = 2 and uses a difference between the current and previous observation as the second input.')
    parser.add_argument('--tests', type=int, default=10, help='The number of times to simulate the system after training.')
    parser.add_argument('--forward_predict', default=False, help='Enables forward predictions.')
    parser.add_argument('--initializer', type=str, default='orthogonal', help='The initializer to use for the weights. Valid values are: {}'.format(initializers))
    args = parser.parse_args()

    if not args.disable_critic: TASKS.append('critic')
    if not args.disable_actor:  TASKS.append('actor')
    if args.reconstruct:        TASKS.append('reconstruct')
    if args.forward_predict:    TASKS.append('forward_predict')
    if len(TASKS) == 0: raise ValueError('No task specified.')
    
    if args.max_ep_steps > 0:
        MAX_EP_STEPS = args.max_ep_steps
    else:
        raise ValueError('max_ep_steps must be positive.')

    GAME = args.game
    env = gym.make(GAME).unwrapped
    env = TimeLimit(env, max_episode_steps=MAX_EP_STEPS)
    N_S = env.observation_space.shape[0]
    CONTINUOUS = not hasattr(env.action_space, 'n')
    if CONTINUOUS:
        N_A = env.action_space.shape[0]
        A_BOUND = [env.action_space.low, env.action_space.high]
    else:
        N_A = env.action_space.n

    if args.entropy_beta > 0:
        ENTROPY_BETA = args.entropy_beta
    else:
        raise ValueError("entropy_beta must be greater than 0.")

    if args.hard_share not in ['equal_params', 'equal_hiddens', 'none']:
        raise ValueError("Hard sharing options are 'equal_params' and 'equal_hiddens'.")

    if args.hard_share == 'none':
        args.hard_share = None

    soft_share = {
        'actor': args.soft_share_actor,
        'critic': args.soft_share_critic,
        'forward_predict': args.soft_share_forward_predict,
        'reconstruct': args.soft_share_reconstruct,
    }
    if args.soft_share > 0:
        soft_share['actor'] = args.soft_share
        soft_share['critic'] = args.soft_share
        soft_share['forward_predict'] = args.soft_share
        soft_share['reconstruct'] = args.soft_share

    gradient_clip = {
        'actor': args.gradient_clip_actor,
        'critic': args.gradient_clip_critic,
        'forward_predict': args.gradient_clip_forward,
        'reconstruct': args.gradient_clip_reconstruct,
    }
    if args.gradient_clip > 0:
        gradient_clip['actor'] = args.gradient_clip
        gradient_clip['critic'] = args.gradient_clip
        gradient_clip['forward_predict'] = args.gradient_clip
        gradient_clip['reconstruct'] = args.gradient_clip

    lr = {
        'actor': args.lr_a,
        'critic': args.lr_c,
        'forward_predict': args.lr_f,
        'reconstruct': args.lr_r,
    }
    if args.lr > 0:
        lr['actor'] = args.lr,
        lr['critic'] = args.lr,
        lr['forward_predict'] = args.lr,
        lr['reconstruct'] = args.lr,

    MAX_GLOBAL_EP = args.max_global_ep
    if MAX_GLOBAL_EP < 1:
        raise ValueError('max_global_ep must be a postive integer.')

    UPDATE_GLOBAL_ITER = args.update_global_iter
    if UPDATE_GLOBAL_ITER < 1:
        raise ValueError('update_global_iter must be a postive integer.')
    '''
    if args.log:
        logger.configure('tmp')
        print("logger dir: ", logger.get_dir())

    if args.debug:
        logger.set_level(logger.DEBUG)
    '''

    if args.optimizer == 'rmsprop':
        optimizer_class = tf.train.RMSPropOptimizer
    elif args.optimizer == 'adam':
        optimizer_class = tf.train.AdamOptimizer
    elif args.optimizer == 'adagrad':
        optimizer_class = tf.train.AdagradOptimizer

    if args.stack < 1:
        raise ValueError('Number of frames ({}) to stack must be positive.'.format(args.stack))

    if args.obs_diff:
        if args.stack <= 2:
            args.stack = 2
        else:
            raise ValueError('stack should be less than or equal to 2 for use with obs_diff.')

    if args.image_shape is not None:
        IMAGE_SHAPE = tuple(map(int, args.image_shape[0].split(','))) + (3,) # Add the number of channels which will always be 3 for RGB images

        env_reset_fn = env_reset_img
        env_get_obs_fn = env_get_img
        env_step_fn = env_step_img
    else:
        env_reset_fn = env_reset_obs
        env_get_obs_fn = env_get_obs
        env_step_fn = env_step_obs

    if args.hold < 1:
        raise ValueError('Hold must be greater than or equal to one.')

    FORWARD_PREDICT = args.forward_predict
    if FORWARD_PREDICT not in [False, 'mse', 'vae']:
        raise ValueError('forward_predict must be either \'mse' or 'vae\'.')

    RECONSTRUCT = args.reconstruct
    if RECONSTRUCT not in [False, 'mse', 'vae']:
        raise ValueError('reconstruct must be either \'mse' or 'vae\'.')

    if args.debug_worker:
        N_WORKERS = 1

    if args.tests < 0:
        raise ValueError("tests must be greater than or equal to 0.")

    if args.initializer not in initializers:
        raise ValueError("initializer must be one of: {}".format(initializers))
    elif args.initializer == 'standard_normal':
        w_init = tf.random_normal_initializer(0., 1.)
    else: # orthogonal
        if args.initializer is not 'orthogonal':
            print('Unknown initializer ({}) defaulting to orthogonal initialization.')
        w_init = tf.orthogonal_initializer()

    print("game: ", GAME)
    print("continuous ", CONTINUOUS)
    print("actions: ", N_A)
    print("observations: ", N_S)
    print("hard_share: ", args.hard_share)
    print("soft_share_actor: ", soft_share['actor'])
    print("soft_share_critic: ", soft_share['critic'])
    print("soft_share_forward_predict: ", soft_share['forward_predict'])
    print("soft_share_reconstruct: ", soft_share['reconstruct'])
    print("gradient clip. actor: ", gradient_clip['actor'])
    print("gradient clip, critic: ", gradient_clip['critic'])
    print("gradient clip, forward: ", gradient_clip['forward_predict'])
    print("gradient clip reconstruct: ", gradient_clip['reconstruct'])
    print("learning rate, actor: ", lr['actor'])
    print("learning rate, critic: ", lr['critic'])
    print("learning rate, forward: ", lr['forward_predict'])
    print("learning rate, reconstruct: ", lr['reconstruct'])
    print("optimizer_class: ", optimizer_class)
    print("max_global_ep: ", MAX_GLOBAL_EP)
    print("update_global_iter: ", UPDATE_GLOBAL_ITER)
    print("max_ep_steps: ", MAX_EP_STEPS)
    print("image_shape: ", IMAGE_SHAPE)
    print("obs_diff: ", args.obs_diff)
    print("reconstruct: ", args.reconstruct)
    print("stack: ", args.stack)
    print("hold: ", args.hold)
    print("tests: ", args.tests)
    print("initializer: ", args.initializer)

    return args, env, soft_share, gradient_clip, lr, optimizer_class, w_init

def run_tests(n_tests, randomize_start=False, start_state=np.array([np.pi, 0])):
    ep_rs = []
    buffer_a = []
    buffer_r = []
    for idx in range(n_tests):
        s, _ = env_reset_fn(env)
        if not randomize_start:
            env.env.state = start_state
            s = env_get_obs_fn(env)

        env.render()
        buffer_s = [s]*(args.stack-1)
        tidx = 0
        ep_r = 0
        done = False
        while tidx < 1000 and not done:
            buffer_s.append(s)
            a = GLOBAL_AC.choose_action(buffer_s[-args.stack:])
            buffer_a.append(a)
            #s_, r, done, info = env.step(np.array([a]))
            s_, r, done, info = env_step_fn(env, np.array([a]))
            buffer_r.append(r)
            ep_r += r
            env.render()
            s = s_
            tidx += 1
        print('ep_r: ', ep_r)
        ep_rs.append(ep_r)

    reconstructions = []
    if 'reconstruct' in TASKS:
        reconstructions, buffer_s_ = GLOBAL_AC.reconstruct(buffer_s)

        print('Mean reconstruction error: ',np.mean( buffer_s - reconstructions))

        if IMAGE_SHAPE is not None:
            plt.imshow(np.squeeze(reconstructions[0]))
            plt.show()
        
    return ep_rs, buffer_s, buffer_a, buffer_r, reconstructions


if __name__ == "__main__":
    args, env, soft_share,  gradient_clip, lr, optimizer_class, w_init = parse_args()
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        for task in TASKS:
            OPT[task] = optimizer_class(lr[task], name=task+'_opt')

        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, w_init, soft_share, gradient_clip, hard_share=args.hard_share, stack=args.stack, batch_normalize=args.batch_normalize, obs_diff=args.obs_diff)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, w_init, GLOBAL_AC, soft_share, gradient_clip, hard_share=args.hard_share, debug=args.debug, stack=args.stack, hold=args.hold, batch_normalize=args.batch_normalize, obs_diff=args.obs_diff))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    if N_WORKERS == 1 or args.debug_worker:
        workers[0].work()
    else:
        tic = time.clock()
        worker_threads = []
        for worker in workers:
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)
        toc = time.clock()
        print('train time:', toc - tic)

    plt.subplot(2, 1, 1)
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(len(GLOBAL_R)), GLOBAL_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    if args.log:
        name = 'plot_'+str(MAX_GLOBAL_EP)+'_sharing_'
        if args.hard_share is not None:
            name += 'hard'
        elif soft_sharing_coeff['actor'] > 0. or soft_sharing_coeff['critic'] > 0.:
            name += 'soft'
        else:
            name += 'none'
        name += '_lra_'+str(lr_a)+'_lrc_'+str(lr_c)+'.png'
        plt.savefig()
    else:
        plt.show()

        ep_rs, buffer_s, buffer_a, buffer_r, reconstructions = run_tests(args.tests)
