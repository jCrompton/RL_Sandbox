import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym

def discount_rewards(rewards, gamma=0.99):
    # Takes 1D array of rewards (floats) and computes the discounted reward
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(xrange(0,rewards.size)):
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

class Agent():
    def __init__(self, learning_rate, state_size, action_size, hidden_state_size):
        self.input = tf.placeholder(shape=[None,state_size], dtype=tf.float32)
        hidden = slim.fully_connected(self.input, hidden_state_size, biases_initializer=None, activation_fn=tf.nn.relu)
        self.output = slim.fully_connected(hidden,action_size,activation_fn=tf.nn.softmax, biases_initializer=None)
        self.chosen_action = tf.argmax(self.output,1)

        self.reward_holder = tf.placeholder(shape[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtypetf.int32)

        self.indices = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs)*self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx,var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float
