import tensorflow as tf
import numpy as np

import gym

import random
import time
from collections import deque

import warnings

warnings.filerwarnings('ignore')

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork', hidden_layer_size = 120):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):
            # Create placeholders
            # Where *state_size just unfolds the state_size array e.g.: if state_size was [5,5] then we'd write [None, 5, 5]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None, action_size], name='actions')

            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.fc = tf.layers.dense(inputs=self.inputs_, units=hidden_layer_size, activation=tf.nn.elu, name='fc1')
            self.output = tf.layers.dense(inputs=self.fc, units=action_size)

            # Q is predicted Q
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.action_), axis=1)

            # The loss is the square diff between predicted Q and target Q
            self.loss = tf.reduce_mean(tf.square(self.Q - self.target_Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

class Memory():
    def __init__(self, env, max_size, pretrain_length, stacked_frames):
        self.buffer = deque(maxlen=max_size)
        self._prepopulate(env, pretrain_length, stacked_frames)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

    def _prepopulate(env, pretrain_length, stacked_frames):
        for i in range(pretrain_length):
            if i == 0:
                state = env.reset()
                state, stacked_frames = stack_frames(stacked_frames, state, True)

            # Random action
            action = env.action_space.sample()

            next_state, reward, done, _  = env.step(action)

            if done:
                memory.add((state, action, reward, next_state, done))

                # Reset
                state = env.reset()

                state, stacked_frames = stack_frames(stacked_frames, state, True)

            else:
                next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                memory.add((state, action, reward, next_state, done))
                state = next_state



def __main__(stack_size=4, env_name='CartPole-v1', learning_rate=0.0002, gamma=0.95, batch_size=64, total_episodes=500, memory_size=1000000):
    pretrain_length = batch_size



