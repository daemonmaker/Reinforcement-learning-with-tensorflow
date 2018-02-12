"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0

TODO
- Add VAE loss to reconstructions
- Add cos activation
- Add forward predictions
- Add evaluation at every X epochs
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

#env = gym.make(GAME)
N_S = 0 #env.observation_space.shape[0]
N_A = 0 #env.action_space.n
A_BOUND = []
CONTINUOUS = False


def summarize_weights():
    temp = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print(len(temp))
    for layer in temp:
        print(layer)

class ACNet(object):
    def __init__(self, scope, globalAC=None, hard_share=None, soft_sharing_coeff_actor=0.0, soft_sharing_coeff_critic=0.0, soft_sharing_coeff_reconstruct=0.0, gradient_clip_actor=0.0, gradient_clip_critic=0.0, gradient_clip_reconstruct=0.0, image_shape=None, stack=1, reconstruct=False, batch_normalize=False, obs_diff=False):
        self.hard_share = hard_share
        self.image_shape = image_shape
        self.stack = stack
        self.reconstruct = reconstruct
        self.batch_normalize = batch_normalize
        self.obs_diff = obs_diff

        def input_placeholders():
            if self.image_shape is not None:
                s = tuple([tf.placeholder(tf.float32, [None, ] + list(self.image_shape), 'S') for _ in range(self.stack)])
            else:
                s = tuple([tf.placeholder(tf.float32, [None, N_S], 'S') for _ in range(self.stack)])
            return s

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = input_placeholders()
                #self.a_params, self.c_params, self.r_params = self._build_net(scope)
                self.outputs, self.params = self._build_net(scope)
                self.a_params = self.params['a_params']
                self.c_params = self.params['c_params']
                self.r_params = self.params['r_params']
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = input_placeholders()
                action_shape = [None, N_A] if CONTINUOUS else [None, ]
                a_dtype = tf.float32 if CONTINUOUS else tf.int32
                self.a_his = tf.placeholder(a_dtype, action_shape, 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                #self.a_prob, self.v, self.reconstruction, self.a_params, self.c_params, self.r_params = self._build_net(scope)
                self.outputs, self.params = self._build_net(scope)
                self.a_prob = self.outputs['a_prob']
                self.v = self.outputs['v']
                self.reconstruction = self.outputs['reconstruct']
                self.a_params = self.params['a_params']
                self.c_params = self.params['c_params']
                self.r_params = self.params['r_params']
                if CONTINUOUS:
                    mu = self.a_prob[0]
                    sigma = self.a_prob[1]

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.t_td = td
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                    if soft_sharing_coeff_actor > 0:
                        self.c_loss += soft_sharing_coeff_critic*tf.nn.l2_loss(self.l_a - self.l_c)

                if CONTINUOUS:
                    with tf.name_scope('wrap_a_out'):
                        mu, sigma = mu*A_BOUND[1], sigma + 1e-4

                    normal_dist = tf.distributions.Normal(mu, sigma)

                with tf.name_scope('a_loss'):
                    if CONTINUOUS:
                        log_prob = normal_dist.log_prob(self.a_his)
                    else:
                        log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)

                    exp_v = log_prob * tf.stop_gradient(td)

                    entropy_beta = ENTROPY_BETA
                    if CONTINUOUS:
                        entropy = normal_dist.entropy()  # encourage exploration
                        #entropy_beta = np.abs(np.random.randn(1))*ENTROPY_BETA
                    else:
                        entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                                 axis=1, keep_dims=True)  # encourage exploration
                    print('entropy_beta: ', entropy_beta)
                    self.t_log_prob = log_prob
                    self.t_entropy = entropy
                    self.t_exp_v = exp_v
                    self.exp_v = entropy_beta * entropy + exp_v
                    self.t_exp_v2 = self.exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    if soft_sharing_coeff_critic > 0:
                        self.a_loss += soft_sharing_coeff_actor*tf.nn.l2_loss(self.l_a - self.l_c)

                if CONTINUOUS:
                    with tf.name_scope('choose_a'):  # use local params to choose action
                        self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), A_BOUND[0], A_BOUND[1])

                if self.reconstruct:
                    with tf.name_scope('reconstruction_loss'):
                        self.r_loss = tf.reduce_mean(tf.losses.mean_squared_error(self.s[-1], self.reconstruction))
                        #self.r_loss = tf.reduce_mean(tf.keras.backend.binary_crossentropy(self.s[-1], self.reconstruction))
                        if soft_sharing_coeff_reconstruct > 0:
                            self.r_loss += soft_sharing_coeff_reconstruct*tf.nn.l2_loss(self.l_a - self.l_r)
                            self.r_loss += soft_sharing_coeff_reconstruct*tf.nn.l2_loss(self.l_c - self.l_r)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    if gradient_clip_actor > 0:
                        self.a_grads, _ = tf.clip_by_global_norm(self.a_grads, gradient_clip_actor)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)
                    if gradient_clip_critic > 0:
                        self.c_grads, _ = tf.clip_by_global_norm(self.c_grads, gradient_clip_critic)
                    if self.reconstruct:
                        self.r_grads = tf.gradients(self.r_loss, self.r_params)
                        if gradient_clip_reconstruct > 0:
                            self.r_grads, _ = tf.clip_by_global_norm(self.r_grads, gradient_clip_reconstruct)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                    if self.reconstruct:
                        self.pull_r_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.r_params, globalAC.r_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))
                    if self.reconstruct:
                        self.update_r_op = OPT_R.apply_gradients(zip(self.r_grads, globalAC.r_params))

    def _batch_normalize(self, inputs):
        if self.batch_normalize:
            return tf.layers.batch_normalization(inputs)
        else:
            return inputs

    def _build_obs_processor(self, obs, w_init, n_out, reuse, scope):
        with tf.variable_scope(scope):
            if self.image_shape is not None:
                conv1 = tf.layers.conv2d(self._batch_normalize(obs), 16, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c1')
                pool1 = tf.layers.max_pooling2d(conv1, 2, 1, name='p1')
                conv2 = tf.layers.conv2d(self._batch_normalize(pool1), 16, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c2')
                pool2 = tf.layers.max_pooling2d(conv2, 2, 1, name='p2')
                conv3 = tf.layers.conv2d(self._batch_normalize(pool2), 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c3')
                conv4 = tf.layers.conv2d(conv3, 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c4')
                pool4 = tf.layers.max_pooling2d(conv4, 2, 1, name='p3')
                conv5 = tf.layers.conv2d(self._batch_normalize(pool4), 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c5')
                conv6 = tf.layers.conv2d(conv5, 32, 3, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='c6')
                flattened_conv = tf.layers.flatten(conv6, name='flattened_conv')
                obs = tf.layers.dense(flattened_conv, 512, tf.nn.relu6, kernel_initializer=w_init, reuse=reuse, name='fc')
            final = tf.layers.dense(obs, n_out, tf.nn.relu6, kernel_initializer=w_init, reuse=reuse, name='processed_obs')
        return final

    def _process_inputs(self, obses, w_init, n_out, reuse=False, share_processor=False):
        if share_processor:
            scope_template = 'input_processor'
        else:
            scope_template = 'input_processor_{}'
        processed_inputs = []
        for idx, obs in enumerate(obses):
            processed_inputs.append(self._build_obs_processor(obs, w_init, n_out, reuse, scope=scope_template.format(idx)))
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

    def _build_deconv(self, inputs, w_init, reuse=False):
        with tf.variable_scope('deconvs'):
            temp = tf.layers.dense(inputs, 400, reuse=reuse, name='temp')
            inputs = tf.reshape(temp, [-1, 5, 5, 16], name='reshaped_flat')
            deconv1 = tf.layers.conv2d_transpose(inputs, 8, 3, strides=(2,2), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d1')
            deconv2 = tf.layers.conv2d_transpose(deconv1, 16, 3, strides=(2,2), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d2')
            deconv3 = tf.layers.conv2d_transpose(deconv2, 32, 5, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d3')
            inputs = tf.layers.conv2d_transpose(deconv3, 1, 8, strides=(1,1), padding='same', activation=tf.nn.relu, kernel_initializer=w_init, bias_initializer=tf.constant_initializer(0.1), reuse=reuse, name='d4')
            #inputs = deconv2
        return inputs

    def _build_hard_share(self, scope, w_init, n_out = 300, share_input_processor=False):
        with tf.variable_scope('input_processor'):
            l_p = self._process_inputs(self.s, w_init, n_out, share_processor=share_input_processor)

        with tf.variable_scope('actor'):
            if CONTINUOUS:
                mu = tf.layers.dense(l_p, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(l_p, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                a_prob = (mu, sigma)
            else:
                a_prob = tf.layers.dense(l_p, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        with tf.variable_scope('critic'):
            v = tf.layers.dense(l_p, 1, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('reconstruct'):
            if self.image_shape is not None:
                reconstruct = self._build_deconv(l_p, w_init, reuse=False) # TODO DWEBB Address sharing of deconv parameters...
            else:
                reconstruct = tf.layers.dense(l_p, N_S, kernel_initializer=w_init,  name='r')

        outputs = {'a_prob': a_prob, 'v': v, 'reconstruct': reconstruct}
        conv_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/input_processor')
        params = {
            'a_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor') + conv_params,
            'c_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic') + conv_params,
            'r_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct') + conv_params,
        }

        return outputs, params

    def _build_soft_share(self, scope, w_init, n_hiddens, share_input_processor=False):
        # TODO DWEBB Store links to the different layers so they can be penalized with soft sharing schemes, e.g. via the regularization in the loss.
        with tf.variable_scope('actor'):
            self.l_a = self._process_inputs(self.s, w_init, n_hiddens['a'], share_processor=share_input_processor)
            if CONTINUOUS:
                mu = tf.layers.dense(self.l_a, N_A, tf.nn.tanh, kernel_initializer=w_init, name='mu')
                sigma = tf.layers.dense(self.l_a, N_A, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
                a_prob = (mu, sigma)
            else:
                a_prob = tf.layers.dense(self.l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')

        with tf.variable_scope('critic'):
            self.l_c = self._process_inputs(self.s, w_init, n_hiddens['c'], share_processor=share_input_processor)
            v = tf.layers.dense(self.l_c, 1, kernel_initializer=w_init, name='v')  # state value

        with tf.variable_scope('reconstruct'):
            self.l_r = self._process_inputs(self.s, w_init, n_hiddens['r'], share_processor=share_input_processor)
            if self.image_shape is not None:
                reconstruct = self._build_deconv(self.l_r, w_init, reuse=False)
            else:
                reconstruct = tf.layers.dense(self.l_r, N_S, kernel_initializer=w_init, name='r')

        outputs = {'a_prob': a_prob, 'v': v, 'reconstruct': reconstruct}
        params = {
            'a_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor'),
            'c_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic'),
            'r_params': tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct'),
        }

        return outputs, params

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)

        n_hiddens = {'a': 200, 'c': 100, 'r': 100}
        if self.hard_share is not None:
            n_out = np.sum(list(n_hiddens.values()))
            if not self.reconstruct:
                n_out -= n_hiddens['r']
            outputs, params = self._build_hard_share(scope, w_init, n_out=n_out, share_input_processor=True)
        else:
            outputs, params = self._build_soft_share(scope, w_init, n_hiddens=n_hiddens, share_input_processor=False)

        return outputs, params

    def _build_net_old(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        inputs = self.s
        s_params = []
        reconstruct = ''
        if self.hard_share is not None:
            if self.image_shape is not None:
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
                if self.reconstruct:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if self.image_shape is not None:
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
                if self.reconstruct:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if self.image_shape is not None:
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
                if self.reconstruct:
                    with tf.variable_scope('reconstruct'):
                        l_r = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                        if self.image_shape is not None:
                            reconstruct = self._build_deconv(l_r, w_init)
                        else:
                            reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')  # state value

            s_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/shared')
        else:
            if self.image_shape is not None:
                concat_inputs = self._build_conv(inputs, w_init)
            else:
                concat_inputs = tf.concat(inputs, 1)

            with tf.variable_scope('actor'):
                '''
                if self.image_shape is not None:
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
                if self.image_shape is not None:
                    concat_inputs = self._build_conv(inputs, w_init)
                else:
                    concat_inputs = tf.concat(inputs, 1)
                '''
                l_c = tf.layers.dense(concat_inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                #l_c = tf.layers.dense(l_c, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc1')
                self.l_c = l_c
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            if self.reconstruct:
                with tf.variable_scope('reconstruct'):
                    '''
                    if self.image_shape is not None:
                        concat_inputs = self._build_conv(inputs, w_init)
                    else:
                        concat_inputs = tf.concat(inputs, 1)
                    '''
                    l_r = tf.layers.dense(concat_inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lr')
                    l_r = tf.layers.dense(l_r, 100, tf.nn.relu6, kernel_initializer=w_init, name='lr1')
                    self.l_r = l_r
                    if self.image_shape is not None:
                        reconstruct = self._build_deconv(l_r, w_init)
                    else:
                        reconstruct = tf.layers.dense(l_r, N_S, kernel_initializer=w_init, name='r')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        if self.image_shape is not None:
            i_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/convs')
            if self.reconstruct:
                i_params += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/deconvs')
        else:
            i_params = []
        if self.reconstruct:
            r_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/reconstruct')
        else:
            r_params = []
        return a_prob, v, reconstruct, a_params + s_params + i_params, c_params + s_params + i_params, r_params + s_params + i_params

    def get_stats(self, feed_dict):
        return SESS.run([self.a_loss, self.c_loss, self.t_td, self.c_loss, self.t_log_prob, self.t_exp_v, self.t_entropy, self.t_exp_v2, self.a_loss, self.a_grads, self.c_grads], feed_dict)

    def update_global(self, feed_dict):  # run by a local
        ops = [self.c_loss, self.a_loss, self.t_exp_v, self.update_a_op, self.update_c_op]
        if self.reconstruct:
            ops.append(self.r_loss)
            ops.append(self.update_r_op)
        results = SESS.run(ops, feed_dict)  # local grads applies to global net
        c_loss, a_loss, entropy = results[:3]
        if self.reconstruct:
            r_loss = results[5]
        else:
            r_loss = 0
        return a_loss, c_loss, r_loss, entropy[0, 0]

    def pull_global(self):  # run by a local
        ops = [self.pull_a_params_op, self.pull_c_params_op]
        if self.reconstruct:
            ops.append(self.pull_r_params_op)
        SESS.run(ops)

    def choose_action(self, s):  # run by a local
        temp = [obs[np.newaxis, :] for obs in s]
        feed_dict = {var: obs for var, obs in zip(self.s, temp)}
        if CONTINUOUS:
            action = np.squeeze(SESS.run(self.A, feed_dict=feed_dict)[0])
            #action = np.squeeze(SESS.run(self.A, feed_dict=feed_dict))
        else:
            prob_weights = SESS.run(self.a_prob, feed_dict=feed_dict)
            action = np.random.choice(range(prob_weights.shape[1]),
                                      p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC, hard_share=None, soft_sharing_coeff_actor=0.0, soft_sharing_coeff_critic=0.0, soft_sharing_coeff_reconstruct=0.0, gradient_clip_actor=0.0, gradient_clip_critic=0.0, gradient_clip_reconstruct=0.0, debug=False, image_shape=None, stack=1, hold=1, reconstruct=False, batch_normalize=False, obs_diff=False):
        self.env = gym.make(GAME).unwrapped
        self.env = TimeLimit(self.env, max_episode_steps=MAX_EP_STEPS)
        self.name = name
        self.AC = ACNet(name, globalAC, hard_share=hard_share, soft_sharing_coeff_actor=soft_sharing_coeff_actor, soft_sharing_coeff_critic=soft_sharing_coeff_critic, soft_sharing_coeff_reconstruct=soft_sharing_coeff_reconstruct, gradient_clip_actor=gradient_clip_actor, gradient_clip_critic=gradient_clip_critic, gradient_clip_reconstruct=gradient_clip_reconstruct, image_shape=image_shape, stack=stack, reconstruct=reconstruct, batch_normalize=batch_normalize, obs_diff=obs_diff)
        self.debug = debug
        self.image_shape = image_shape
        self.stack = stack
        self.hold = hold
        self.reconstruct = reconstruct

    def work(self):
        def get_img(fn, *args):
            img_lock.acquire()
            results = fn(*args)
            if CONTINUOUS and type(self.env.env) is gym.envs.classic_control.pendulum.PendulumEnv:
                img = self.env.render(mode='rgb_array_no_arrow')
            else:
                img = self.env.render(mode='rgb_array')
            img_lock.release()
            img = rgb2grey(img)
            img = resize(img, self.image_shape)
            return img, results

        def env_reset_obs():
            s = self.env.reset()
            #if type(self.env.env) is gym.envs.classic_control.pendulum.PendulumEnv:
            #    self.env.env.state = np.array([np.pi, 0])
            #    s = self.env.env._get_obs()
            return s, None

        def env_reset_img():
            img, results = get_img(env_reset_obs)
            return img, results

        def env_step_obs(a):
            return self.env.step(a)

        def env_step_img(a):
            img, results = get_img(env_step_obs, a)
            return img, results[1], results[2], results[3]

        if self.image_shape is not None:
            env_reset_fn = env_reset_img
            env_step_fn = env_step_img
        else:
            env_reset_fn = env_reset_obs
            env_step_fn = env_step_obs

        global GLOBAL_RUNNING_R, GLOBAL_R, GLOBAL_EP, MAX_GLOBAL_EP
        total_step = 1
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s, _ = env_reset_fn()

            buffer_s = [s]*(self.stack-1)
            ep_r = 0
            action_count = 0
            for ep_t in range(MAX_EP_STEPS):
            #while True:
                buffer_s.append(s)
                if action_count % self.hold == 0:
                    a = self.AC.choose_action(buffer_s[-self.stack:])
                #if self.name == 'W_0':
                #    print('ep_r: ', ep_r, '\taction: ', a)
                action_count += 1
                s_, r, done, info = env_step_fn(np.array([a])) # HACK

                if CONTINUOUS:
                    done = True if ep_t == MAX_EP_STEPS - 1 else False
                elif done: r = -2000 #-5

                ep_r += r
                buffer_a.append(a)
                if CONTINUOUS:
                    buffer_r.append((r+8)/8)
                else:
                    buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    #import ipdb; ipdb.set_trace()

                    if done:
                        v_s_ = 0   # terminal
                    else:
                        obs_hist = buffer_s[-self.stack:]
                        feed_dict = {var: obs[np.newaxis, :] for var, obs in zip(self.AC.s, obs_hist)}
                        v_s_ = SESS.run(self.AC.v, feed_dict=feed_dict)[0, 0]

                    buffer_v_target = []
                    for r in buffer_r[::-1]:    # reverse buffer r
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    if self.image_shape is not None:
                        buffer_s_ = [buffer_s_[np.newaxis, :] for buffer_s_ in buffer_s]
                    else:
                        buffer_s_ = copy.deepcopy(buffer_s)

                    if CONTINUOUS:
                        buffer_a = np.vstack(buffer_a)
                    else:
                        buffer_a = np.array(buffer_a)

                    buffer_v_target = np.vstack(buffer_v_target)

                    obs_columns = [np.vstack(buffer_s_[idx:-(self.stack-(idx+1))]) for idx in range(self.stack-1)]
                    obs_columns.append(np.vstack(buffer_s_[self.stack-1:]))
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
                    feed_dict = {var: obs for var, obs in zip(self.AC.s, obs_columns)}
                    feed_dict[self.AC.a_his] = buffer_a
                    feed_dict[self.AC.v_target] = buffer_v_target
                    if self.debug and self.name == 'W_0':
                        a_loss, c_loss, t_td, c_loss, t_log_prob, t_exp_v, t_entropy, t_exp_v2, a_loss, a_grads, c_grads = self.AC.get_stats(feed_dict)
                        #print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss.shape, " ", c_loss, "\ttd: ", t_td.shape, " ", t_td, "\tlog_prob: ", t_log_prob.shape, " ", t_log_prob, "\texp_v: ", t_exp_v.shape, " ", t_exp_v, "\tentropy: ", t_entropy.shape, " ", t_entropy, "\texp_v2: ", t_exp_v2.shape, " ", t_exp_v2, "\ta_grads: ", [np.sum(weights) for weights in a_grads], "\tc_grads: ", [np.sum(weights) for weights in c_grads])
                        print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss)
                    c_loss, a_loss, r_loss, entropy = self.AC.update_global(feed_dict)

                    buffer_s = buffer_s[-(self.stack-1):] if self.stack > 1 else []
                    buffer_a, buffer_r = [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    GLOBAL_R.append(ep_r)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        if CONTINUOUS:
                            GLOBAL_RUNNING_R.append(0.9 * GLOBAL_RUNNING_R[-1] + 0.1 * ep_r)
                        else:
                            GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)

                    print(self.name, ' Ep: ', GLOBAL_EP, ' | Ep_r av: ', GLOBAL_RUNNING_R[-1], ' | Ep_r: ', ep_r, end='')
                    if self.reconstruct:
                        print(' | r_loss: ', r_loss, end='')
                    print('')
                    '''
                    log_lock.acquire()
                    logger.record_tabular("global_ep", GLOBAL_EP)
                    logger.record_tabular("name", self.name)
                    logger.record_tabular("ep_r", ep_r)
                    logger.record_tabular("ep_r_weighted", GLOBAL_RUNNING_R[-1])
                    logger.record_tabular("c_loss", c_loss)
                    logger.record_tabular("a_loss", a_loss)
                    if self.reconstruct:
                        logger.record_tabular("r_loss", r_loss)
                    logger.record_tabular("entropy", entropy)
                    logger.dump_tabular()
                    log_lock.release()
                    '''

                    GLOBAL_EP += 1
                    break


def parse_args():
    global GAME, A_BOUND, ENTROPY_BETA, MAX_EP_STEPS, UPDATE_GLOBAL_ITER, MAX_GLOBAL_EP, N_S, N_A, CONTINUOUS

    parser = argparse.ArgumentParser(description='Run A3C on discrete cart-pole.')
    parser.add_argument('--game', default='CartPole-v0', help='Which environment to learn to control.')
    parser.add_argument('--entropy_beta', type=float, default=0.01, help="Value for the entropy beta term.")
    parser.add_argument('--hard_share', type=str, default='none', help="Indicates whether the models should have an equal number of parameters ('equal_params'), an equal number of hidden units ('equal_hiddens'), or no sharing ('none' -- default).")
    parser.add_argument('--soft_share', type=float, default=0.0, help='Enables soft sharing of both actor and critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_actor', type=float, default=0.0, help='Enables soft sharing of actor parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_critic', type=float, default=0.0, help='Enables soft sharing of critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_reconstruct', type=float, default=0.0, help='Enables soft sharing of reconstruction parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--gradient_clip', type=float, default=0.0, help='Enables gradient clipping of actor and critic parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_actor', type=float, default=0.0, help='Enables gradient clipping of actor parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_critic', type=float, default=0.0, help='Enables gradient clipping of critic parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_reconstruct', type=float, default=0.0, help='Enables gradient clipping of reconstruction parameters with the specificied maximum gradient.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enables debugging output.')
    parser.add_argument('--optimizer', default='adagrad', help='Which optimizer to use: rmsprop, adam, adagrad.')
    parser.add_argument('--lr', type=float, default=0.0, help='Sets the learning rate of the actor and critic.')
    parser.add_argument('--lr_a', type=float, default=0.001, help='Sets the learning rate of the actor.')
    parser.add_argument('--lr_c', type=float, default=0.001, help='Sets the learning rate of the critic.')
    parser.add_argument('--lr_r', type=float, default=0.001, help='Sets the learning rate of the reconstruction.')
    parser.add_argument('--log', default=False, action='store_true', help='Enables logging.')
    parser.add_argument('--max_global_ep', type=int, default=500, help='Sets the maximum number of episodes to be executed across all threads.')
    parser.add_argument('--update_global_iter', type=int, default=100, help='How frequently to update the global AC.')
    parser.add_argument('--max_ep_steps', type=int, default=1000, help='The number of time steps per episode before calling the episode done.')
    parser.add_argument('--image_shape', nargs='*', default=None, help='Designates that images shoud be used in lieu of observations and what shpae to use for them.')
    parser.add_argument('--debug_worker', default=False, action='store_true')
    parser.add_argument('--reconstruct', default=False, action='store_true', help='Enables observation reconstruction as an additional learning signal.')
    parser.add_argument('--batch_normalize', default=False, action='store_true', help='Enables batch normalization of the convolutional layers.')
    parser.add_argument('--stack', type=int, default=1, help='Number of observations to use for state.')
    parser.add_argument('--hold', type=int, default=1, help='Number of time steps to hold the control.')
    parser.add_argument('--obs_diff', default=False, action='store_true', help='Requires stack = 2 and uses a difference between the current and previous observation as the second input.')
    args = parser.parse_args()

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

    soft_share_actor = args.soft_share_actor
    soft_share_critic = args.soft_share_critic
    soft_share_reconstruct = args.soft_share_reconstruct
    if args.soft_share > 0:
        soft_share_actor = args.soft_share
        soft_share_critic = args.soft_share
        soft_share_reconstruct = args.soft_share

    gradient_clip_actor = args.gradient_clip_actor
    gradient_clip_critic = args.gradient_clip_critic
    gradient_clip_reconstruct = args.gradient_clip_reconstruct
    if args.gradient_clip > 0:
        gradient_clip_actor = args.gradient_clip
        gradient_clip_critic = args.gradient_clip
        gradient_clip_reconstruct = args.gradient_clip

    lr_a = args.lr_a
    lr_c = args.lr_c
    lr_r = args.lr_r
    if args.lr > 0:
        lr_a = args.lr
        lr_c = args.lr
        lr_r = args.lr

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
        
    image_shape = None
    if args.image_shape is not None:
        image_shape = tuple(map(int, args.image_shape[0].split(','))) + (1,) # Add the number of channels which will always be 1 for grey scale

    if args.hold < 1:
        raise ValueError('Hold must be greater than or equal to one.')

    print("game: ", GAME)
    print("continuous ", CONTINUOUS)
    print("actions: ", N_A)
    print("observations: ", N_S)
    print("hard_share: ", args.hard_share)
    print("soft_share_actor: ", soft_share_actor)
    print("soft_share_critic: ", soft_share_critic)
    print("gradient_clip_actor: ", gradient_clip_actor)
    print("gradient_clip_critic: ", gradient_clip_critic)
    print("gradient_clip_reconstruct: ", gradient_clip_reconstruct)
    print("gradient_clip_reconstruct: ", gradient_clip_reconstruct)
    print("learning rate, actor: ", lr_a)
    print("learning rate, critic: ", lr_c)
    print("learning rate, reconstruct: ", lr_r)
    print("optimizer_class: ", optimizer_class)
    print("max_global_ep: ", MAX_GLOBAL_EP)
    print("update_global_iter: ", UPDATE_GLOBAL_ITER)
    print("max_ep_steps: ", MAX_EP_STEPS)
    print("image_shape: ", image_shape)
    print("obs_diff: ", args.obs_diff)
    print("reconstruct: ", args.reconstruct)
    print("stack: ", args.stack)
    print("hold: ", args.hold)

    return args, env, soft_share_actor, soft_share_critic, soft_share_reconstruct, gradient_clip_actor, gradient_clip_critic, gradient_clip_reconstruct, lr_a, lr_c, lr_r, optimizer_class, image_shape


if __name__ == "__main__":
    args, env, soft_share_actor, soft_share_critic, soft_share_reconstruct,  gradient_clip_actor, gradient_clip_critic, gradient_clip_reconstruct, lr_a, lr_c, lr_r, optimizer_class, image_shape = parse_args()
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = optimizer_class(lr_a, name='actor_opt')
        OPT_C = optimizer_class(lr_c, name='critic_opt')
        if args.reconstruct:
            OPT_R = optimizer_class(lr_r, name='reconstruct_opt')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, hard_share=args.hard_share, image_shape=image_shape, stack=args.stack, reconstruct=args.reconstruct, batch_normalize=args.batch_normalize, obs_diff=args.obs_diff)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, hard_share=args.hard_share, soft_sharing_coeff_actor=soft_share_actor, soft_sharing_coeff_critic=soft_share_critic, soft_sharing_coeff_reconstruct=soft_share_reconstruct, gradient_clip_actor=gradient_clip_actor, gradient_clip_critic=gradient_clip_critic, gradient_clip_reconstruct=gradient_clip_reconstruct, debug=args.debug, image_shape=image_shape, stack=args.stack, hold=args.hold, reconstruct=args.reconstruct, batch_normalize=args.batch_normalize, obs_diff=args.obs_diff))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    if args.debug_worker:
        workers[0].work()
        exit

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
        elif soft_sharing_coeff_actor > 0. or soft_sharing_coeff_critic > 0.:
            name += 'soft'
        else:
            name += 'none'
        name += '_lra_'+str(lr_a)+'_lrc_'+str(lr_c)+'.png'
        plt.savefig()
    else:
        plt.show()

        for idx in range(10):
            s = env.reset()
            env.state = np.array([0, 0])
            if image_shape is not None:
                img = env.render(mode='rgb_array')
                img = rgb2grey(img)
                s = resize(img, image_shape)
            env.render()
            buffer_s = [s]*(args.stack-1)
            tidx = 0
            ep_r = 0
            done = False
            while tidx < 1000 and not done:
                buffer_s.append(s)
                a = workers[0].AC.choose_action(buffer_s[-args.stack:])
                s_, r, done, info = env.step(np.array([a]))
                ep_r += r
                env.render()
                s = s_
                tidx += 1
            print('ep_r: ', ep_r)
