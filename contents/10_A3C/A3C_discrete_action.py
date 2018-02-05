"""
Asynchronous Advantage Actor Critic (A3C) with discrete action space, Reinforcement Learning.

The Cartpole example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
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
from baselines import logger
from skimage.color import rgb2grey
from skimage.transform import resize
import copy
import sys
from collections import deque


img_lock = threading.Lock()
log_lock = threading.Lock()

GAME = 'CartPole-v0'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 1000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 100 #10
GAMMA = 0.999 # 0.9
ENTROPY_BETA = 0.01 # 0.001
GLOBAL_RUNNING_R = []
GLOBAL_R = []
GLOBAL_EP = 0

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None, hard_share=None, soft_sharing_coeff_actor=0.01, soft_sharing_coeff_critic=0.01, gradient_clip_actor=1.0, gradient_clip_critic=1.0, image_shape=None, stack=1):
        self.hard_share = hard_share
        self.image_shape = image_shape
        self.stack = stack

        def input_placeholders():
            if self.image_shape is not None:
                s = tuple([tf.placeholder(tf.float32, [None, ] + list(self.image_shape), 'S') for _ in range(self.stack)])
            else:
                s = tuple([tf.placeholder(tf.float32, [None, N_S], 'S') for _ in range(self.stack)])
            return s

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = input_placeholders()
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = input_placeholders()
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                self.t_td = td
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))
                    if soft_sharing_coeff_actor > 0:
                        self.c_loss += soft_sharing_coeff_critic*tf.nn.l2_loss(self.l_a - self.l_c)

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    self.t_log_prob = log_prob
                    exp_v = log_prob * tf.stop_gradient(td)
                    self.t_exp_v = exp_v
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.t_entropy = entropy
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.t_exp_v2 = self.exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    if soft_sharing_coeff_critic > 0:
                        self.a_loss += soft_sharing_coeff_actor*tf.nn.l2_loss(self.l_a - self.l_c)

                with tf.name_scope('local_grad'):
                    self.a_grads, _ = tf.clip_by_global_norm(tf.gradients(self.a_loss, self.a_params), gradient_clip_actor)
                    if gradient_clip_actor > 0:
                        self.a_grads, _ = tf.clip_by_global_norm(self.a_grads, gradient_clip_actor)
                    self.c_grads, _ = tf.clip_by_global_norm(tf.gradients(self.c_loss, self.c_params), gradient_clip_critic)
                    if gradient_clip_critic > 0:
                        self.c_grads, _ = tf.clip_by_global_norm(self.c_grads, gradient_clip_critic)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        inputs = self.s
        if self.hard_share is not None:
            if self.image_shape is not None:
                with tf.variable_scope('conv_inputs'):
                    if self.stack == 2:
                        diff = inputs[1] - inputs[0]
                        inputs = tf.concat([inputs[1], diff], -1)
                    else:
                        inputs = tf.concat(inputs, -1)
                    conv1 = tf.layers.conv2d(inputs, 64, 5, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(1.0), name='c1')
                    conv2 = tf.layers.conv2d(conv1, 32, 5, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(1.0), name='c2')
                    conv3 = tf.layers.conv2d(conv2, 16, 3, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(1.0), name='c3')
                    conv4 = tf.layers.conv2d(conv3, 16, 3, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(1.0), name='c4')
                    inputs = tf.layers.flatten(conv4, name='p3')
            else:
                inputs = tf.concat(inputs, 1)
            if self.hard_share == 'equal_params':
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 71, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 16, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            elif False:
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
            else:
                with tf.variable_scope('shared'):
                    l_s = tf.layers.dense(inputs, 200, tf.nn.relu6, kernel_initializer=w_init, name='ls')
                with tf.variable_scope('actor'):
                    l_a = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='la')
                    a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
                with tf.variable_scope('critic'):
                    l_c = tf.layers.dense(l_s, 10, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                    v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value


            s_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/shared')
            a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
            c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
            return a_prob, v, a_params+s_params, c_params+s_params
        else:
            if self.image_shape is not None:
                with tf.variable_scope('conv_inputs'):
                    if self.stack == 2:
                        diff = inputs[1] - inputs[0]
                        inputs = tf.concat([inputs[1], diff], -1)
                    else:
                        inputs = tf.concat(inputs, -1)
                    conv1 = tf.layers.conv2d(inputs, 64, 8, strides=(5,5), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(0.1), name='c1')
                    conv2 = tf.layers.conv2d(conv1, 64, 5, strides=(3,3), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(0.1), name='c2')
                    conv3 = tf.layers.conv2d(conv2, 32, 3, strides=(2,2), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(0.1), name='c3')
                    conv4 = tf.layers.conv2d(conv3, 16, 3, strides=(2,2), padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(), bias_initializer=tf.constant_initializer(0.1), name='c4')
                    inputs = tf.layers.flatten(conv4, name='p3')
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
            else:
                inputs = tf.concat(inputs, 1)
            with tf.variable_scope('actor'):
                l_a = tf.layers.dense(inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='la')
                self.l_a = l_a
                a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
            with tf.variable_scope('critic'):
                l_c = tf.layers.dense(inputs, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
                self.l_c = l_c
                v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        if self.image_shape is not None:
            i_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/conv_inputs')
            a_params += i_params
            c_params += i_params
        return a_prob, v, a_params, c_params

    def get_stats(self, feed_dict):
        return SESS.run([self.a_loss, self.c_loss, self.t_td, self.c_loss, self.t_log_prob, self.t_exp_v, self.t_entropy, self.t_exp_v2, self.a_loss, self.a_grads, self.c_grads], feed_dict)

    def update_global(self, feed_dict):  # run by a local
        a_loss, c_loss, entropy, _, _ = SESS.run([self.c_loss, self.a_loss, self.t_exp_v, self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net
        return a_loss, c_loss, entropy[0, 0]

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        if type(s) is not list:
            import ipdb; ipdb.set_trace()
        temp = [obs[np.newaxis, :] for obs in s ]
        feed_dict = {var: obs for var, obs in zip(self.s, temp)}
        prob_weights = SESS.run(self.a_prob, feed_dict=feed_dict)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC, hard_share=None, soft_sharing_coeff_actor=0.0, soft_sharing_coeff_critic=0.0, gradient_clip_actor=0.0, gradient_clip_critic=0.0, debug=False, max_ep_steps=200, image_shape=None, stack=1):
        self.env = gym.make(GAME).unwrapped
        self.env = TimeLimit(self.env, max_episode_steps=max_ep_steps)
        self.name = name
        self.AC = ACNet(name, globalAC, hard_share=hard_share, soft_sharing_coeff_actor=soft_sharing_coeff_actor, soft_sharing_coeff_critic=soft_sharing_coeff_critic, gradient_clip_actor=gradient_clip_actor, gradient_clip_critic=gradient_clip_critic, image_shape=image_shape, stack=stack)
        self.debug = debug
        self.image_shape = image_shape
        self.stack = stack

    def work(self):
        def get_img(fn, *args):
            img_lock.acquire()
            results = fn(*args)
            img = self.env.render(mode='rgb_array')
            img_lock.release()
            img = rgb2grey(img)
            img = resize(img, self.image_shape)
            return img, results

        def env_reset_obs():
            return self.env.reset()

        def env_reset_img():
            img, _ = get_img(env_reset_obs)
            return img

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
            s = env_reset_fn()
            buffer_s = [s]*self.stack
            ep_r = 0
            while True:
                a = self.AC.choose_action(buffer_s[-self.stack:])
                s_, r, done, info = env_step_fn(a)
                if done: r = -5
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r)

                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net

                    if done:
                        v_s_ = 0   # terminal
                    else:
                        obs_hist = buffer_s[-(self.stack-1):] + [s_,]
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
                    obs_columns = [np.vstack(buffer_s_[idx:-(self.stack-idx)]) for idx in range(self.stack)]
                    buffer_a, buffer_v_target = np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {var: obs for var, obs in zip(self.AC.s, obs_columns)}
                    feed_dict[self.AC.a_his] = buffer_a
                    feed_dict[self.AC.v_target] = buffer_v_target
                    if self.debug and self.name == 'W_0':
                        a_loss, c_loss, t_td, c_loss, t_log_prob, t_exp_v, t_entropy, t_exp_v2, a_loss, a_grads, c_grads = self.AC.get_stats(feed_dict)
                        #print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss.shape, " ", c_loss, "\ttd: ", t_td.shape, " ", t_td, "\tlog_prob: ", t_log_prob.shape, " ", t_log_prob, "\texp_v: ", t_exp_v.shape, " ", t_exp_v, "\tentropy: ", t_entropy.shape, " ", t_entropy, "\texp_v2: ", t_exp_v2.shape, " ", t_exp_v2, "\ta_grads: ", [np.sum(weights) for weights in a_grads], "\tc_grads: ", [np.sum(weights) for weights in c_grads])
                        print("a_loss: ", a_loss.shape, " ", a_loss, "\tc_loss: ", c_loss)
                    c_loss, a_loss, entropy = self.AC.update_global(feed_dict)

                    #import ipdb; ipdb.set_trace()
                    buffer_s, buffer_a, buffer_r = buffer_s[-(self.stack):], [], []
                    self.AC.pull_global()

                s = s_
                total_step += 1
                if done:
                    GLOBAL_R.append(ep_r)
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)

                    log_lock.acquire()
                    logger.record_tabular("global_ep", GLOBAL_EP)
                    logger.record_tabular("name", self.name)
                    logger.record_tabular("ep_r", ep_r)
                    logger.record_tabular("ep_r_weighted", GLOBAL_RUNNING_R[-1])
                    logger.record_tabular("c_loss", c_loss)
                    logger.record_tabular("a_loss", a_loss)
                    logger.record_tabular("entropy", entropy)
                    logger.dump_tabular()
                    log_lock.release()

                    GLOBAL_EP += 1
                    break


def parse_args():
    global MAX_GLOBAL_EP

    parser = argparse.ArgumentParser(description='Run A3C on discrete cart-pole.')
    parser.add_argument('--hard_share', type=str, default='none', help="Indicates whether the models should have an equal number of parameters ('equal_params'), an equal number of hidden units ('equal_hiddens'), or no sharing ('none' -- default).")
    parser.add_argument('--soft_share', type=float, default=0.0, help='Enables soft sharing of both actor and critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_actor', type=float, default=0.0, help='Enables soft sharing of actor parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--soft_share_critic', type=float, default=0.0, help='Enables soft sharing of critic parameters, via L2 loss, with the specificied weight.')
    parser.add_argument('--gradient_clip', type=float, default=0.0, help='Enables gradient clipping of actor and critic parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_actor', type=float, default=0.0, help='Enables gradient clipping of actor parameters with the specificied maximum gradient.')
    parser.add_argument('--gradient_clip_critic', type=float, default=0.0, help='Enables gradient clipping of critic parameters with the specificied maximum gradient.')
    parser.add_argument('--debug', default=False, action='store_true', help='Enables debugging output.')
    parser.add_argument('--optimizer', default='adagrad', help='Which optimizer to use: rmsprop, adam, adagrad.')
    parser.add_argument('--lr', type=float, default=0.0, help='Sets the learning rate of the actor and critic.')
    parser.add_argument('--lr_a', type=float, default=0.001, help='Sets the learning rate of the actor.')
    parser.add_argument('--lr_c', type=float, default=0.001, help='Sets the learning rate of the critic.')
    parser.add_argument('--max_global_ep', type=int, default=500, help='Sets the maximum number of episodes to be executed across all threads.')
    parser.add_argument('--log', default=False, action='store_true', help='Enables logging.')
    parser.add_argument('--max_ep_steps', type=int, default=2000, help='The number of time steps per episode before calling the episode done.')
    parser.add_argument('--stack', type=int, default=1, help='Number of observations to use for state.')
    parser.add_argument('--image_shape', nargs='*', default=None, help='Designates that images shoud be used in lieu of observations and what shpae to use for them.')
    parser.add_argument('--debug_worker', default=False, action='store_true')
    args = parser.parse_args()
    
    if args.hard_share not in ['equal_params', 'equal_hiddens', 'none']:
        raise ValueError("Hard sharing options are 'equal_params' and 'equal_hiddens'.")

    if args.hard_share == 'none':
        args.hard_share = None

    soft_share_actor = args.soft_share_actor
    soft_share_critic = args.soft_share_critic
    if args.soft_share > 0:
        soft_share_actor = args.soft_share
        soft_share_critic = args.soft_share

    gradient_clip_actor = args.gradient_clip_actor
    gradient_clip_critic = args.gradient_clip_critic
    if args.gradient_clip > 0:
        gradient_clip_actor = args.gradient_clip
        gradient_clip_critic = args.gradient_clip

    lr_a = args.lr_a
    lr_c = args.lr_c
    if args.lr > 0:
        lr_a = args.lr
        lr_c = args.lr

    MAX_GLOBAL_EP = args.max_global_ep

    if args.log:
        logger.configure('tmp')
        print("logger dir: ", logger.get_dir())

    if args.debug:
        logger.set_level(logger.DEBUG)

    if args.optimizer == 'rmsprop':
        optimizer_class = tf.train.RMSPropOptimizer
    elif args.optimizer == 'adam':
        optimizer_class = tf.train.AdamOptimizer
    elif args.optimizer == 'adagrad':
        optimizer_class = tf.train.AdagradOptimizer

    if args.stack < 1:
        raise ValueError('Number of frames ({}) to stack must be positive.'.format(args.stack))

    image_shape = None
    if args.image_shape is not None:
        image_shape = tuple(map(int, args.image_shape[0].split(','))) + (1,) # Add the number of channels which will always be 1 for grey scale
        if args.stack < 2:
            args.stack = 2 # Automatically incorporate diffs with previous images

    print("hard_share: ", args.hard_share)
    print("soft_share_actor: ", soft_share_actor)
    print("soft_share_critic: ", soft_share_critic)
    print("gradient_clip_actor: ", gradient_clip_actor)
    print("gradient_clip_critic: ", gradient_clip_critic)
    print("learning rate, actor: ", lr_a)
    print("learning rate, critic: ", lr_c)
    print("max_global_ep: ", MAX_GLOBAL_EP)
    print("optimizer_class: ", optimizer_class)
    print("max_ep_steps: ", args.max_ep_steps)
    print("image_shape: ", image_shape)

    return args, soft_share_actor, soft_share_critic, gradient_clip_actor, gradient_clip_critic, lr_a, lr_c, optimizer_class, image_shape


if __name__ == "__main__":
    args, soft_share_actor, soft_share_critic,  gradient_clip_actor, gradient_clip_critic, lr_a, lr_c, optimizer_class, image_shape = parse_args()
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        OPT_A = optimizer_class(lr_a, name='actor_opt')
        OPT_C = optimizer_class(lr_c, name='critic_opt')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE, hard_share=args.hard_share, image_shape=image_shape, stack=args.stack)  # we only need its params
        workers = []
        # Create worker
        for i in range(N_WORKERS):
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, GLOBAL_AC, hard_share=args.hard_share, soft_sharing_coeff_actor=soft_share_actor, soft_sharing_coeff_critic=soft_share_critic, gradient_clip_actor=gradient_clip_actor, gradient_clip_critic=gradient_clip_critic, debug=args.debug, max_ep_steps=args.max_ep_steps, image_shape=image_shape, stack=args.stack))

    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    if args.debug_worker:
        workers[0].work()
        sys.exit()

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)

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

        env = gym.make(GAME).unwrapped
        env = TimeLimit(env, max_episode_steps=args.max_ep_steps)
        s = env.reset()
        buffer_s = [s]*args.stack
        tidx = 0
        done = False
        while tidx < 1000 and not done:
            a = workers[0].AC.choose_action(buffer_s[-args.stack:])
            env.render()
            s_, r, done, info = env.step(a)
            s = s_
            buffer_s.append(s)
            tidx += 1
