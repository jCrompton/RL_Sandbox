import tensorflow as tf
import tensorflow.layers as layers
import gym
import math

import numpy as np

from skimage import transform
from skimage import data, color

from collections import deque

from PIL import Image

class AutoEncoder:
    def __init__(self, env_name, learning_rate=0.001, batch_size=16, frozen_model_dir="./models/",
                 power_img_size=10, power_latent_size=8, resize_shape=120, noise_factor=0.5, debug=2,
                 pretrain_length=20, max_cache_size=1000000):

        self.env = gym.make(env_name)
        self.debug = debug
        self.architecture = [2 << exponent for exponent in range(power_img_size)]
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.pretrain_length = pretrain_length
        self.cache = deque(maxlen=max_cache_size)
        self.resize_shape = resize_shape
        self.noise_factor = noise_factor
        self.inputs_ = tf.placeholder(tf.float32, (None, *[120, 120, 1]), name='inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, *[120, 120, 1]), name='targets')
        self._prepopulate()
        # encoder
        encoder = self.inputs_
        weights = []
        shapes = []
        print(self.architecture[power_latent_size:])
        for filters in reversed(self.architecture[power_latent_size:]):
            print("Encoder shape : {}".format(encoder.shape))
            n_input = encoder.get_shape().as_list()[3]
            W = tf.Variable(
            tf.random_uniform([
                3,
                3,
                n_input, filters],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
            b = tf.Variable(tf.zeros([filters]))
            conv = tf.nn.conv2d(encoder, W, strides=[1,2,2,1], padding='SAME')
            weights.append(W)
            shapes.append(encoder.get_shape().as_list())
            encoder = tf.add(conv, b)

        self.encoder = encoder
        weights.reverse()
        shapes.reverse()

        decoded = self.encoder
        # decoder
        for i, shape in enumerate(shapes):
            W = weights[i]
            b = tf.Variable(tf.zeros([W.get_shape().as_list()[2]]))
            de_conv = tf.nn.conv2d_transpose(decoded, W, tf.stack([
                tf.shape(self.inputs_)[0], shape[1], shape[2], shape[3]]), strides=[1,2,2,1], padding='SAME')
            decoded = tf.add(de_conv, b)
            print("Decoder shape : {}".format(decoded.shape))

        decoded = tf.nn.sigmoid(decoded)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets_, logits=decoded)
        self.cost = tf.reduce_mean(loss)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.network = decoded

        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def sample(self):
        cache_size = len(self.cache)
        index = np.random.choice(np.arange(cache_size), size=self.batch_size, replace=False)
        return np.array([self.cache[i] for i in index])

    def _prepopulate(self):
        for i in range(self.pretrain_length):
            self.env.reset()
            done = False

            while not done:
                next_state, _, done, _ = self.env.step(self.env.action_space.sample())
                processed_image = self.preprocess_image(next_state)
                self.cache.append(processed_image)

    def preprocess_image(self, frame):
        return transform.resize(color.rgb2gray(frame), [self.resize_shape, self.resize_shape, 1], anti_aliasing=True)

    def add_noise(self, frame):
        frame = frame + self.noise_factor * np.random.randn(*frame.shape)
        return np.clip(frame, 0., 1.)

    def train(self, epochs=10):
        self.sess.run(tf.global_variables_initializer())
        batches = int(np.ceil(len(self.cache) / self.batch_size))
        for epoch in range(epochs):
            avg_cost_epoch = 0.
            for i in range(batches):
                if self.debug > 1:
                    print('Running batch {} with current average cost {}...\n'.format(i, avg_cost_epoch))

                batch = self.sample()
                noisy_batch = self.add_noise(batch)

                batch_cost, _ = self.sess.run([self.cost, self.optimize], feed_dict={self.inputs_: noisy_batch,
                                                                                     self.targets_: batch})
                avg_cost_epoch += batch_cost / batches

            if self.debug > 0:
                print("Epoch: {}/{}...".format(epoch + 1, epochs), "Training loss: {}".format(avg_cost_epoch))
