import argparse

import tensorflow as tf
import numpy as np

import gym

import random
import time
from collections import deque

from skimage import transform
from skimage import data, color

import warnings

warnings.filterwarnings('ignore')

class Agent:
    def __init__(self, env_name='CartPole-v0', total_episodes=10000, batch_size=64, explore_start=1.0,
                 explore_stop=0.01, decay_rate=0.0001, debug=False, pretrain_length=64, 
                 learning_rate=0.0002, memory_size=1000000, gamma=0.95, stack_size=4, debug_rate=6,
                 dense_architecture=[120, 40], dropout=True, activation=tf.nn.relu, dropout_rate=0.25, 
                 tensorboard_dir="scratch/dqn", frozen_model_dir="./models/"):
        env = gym.make(env_name)

        self.env = env
        self.stack_size = stack_size
        self.total_episodes = total_episodes
        self.memory = Memory(env, memory_size, pretrain_length, stack_size)
        self.network = DQNetwork(self.memory.buffer[0][0].shape, env.action_space.n, learning_rate)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.batch_size = batch_size
        self.debug = debug
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = 0
        self.gamma = gamma 
        self.debug_rate = debug_rate

         # Setup tensorboard 
        self.writer = tf.summary.FileWriter(tensorboard_dir)

        # Frozen model dir
        self.frozen_model_dir = "{}{}.ckpt".format(frozen_model_dir, '{}_{}'.format(env_name, total_episodes))

        # Loss
        tf.summary.scalar("DQN_Loss", self.network.loss)
        self.write_op = tf.summary.merge_all()

    def predict_action(self, state):
        exp_exp_tradeoff = np.random.randn()

        explore_probability = self.explore_stop + (self.explore_start - self.explore_stop) * np.exp(-self.decay_rate*self.decay_step)
        
        if (explore_probability > exp_exp_tradeoff):
            # Take random action
            return one_hot_encode_action(self.env.action_space.sample(), self.env.action_space.n), explore_probability

        else:
            # Predict action from DQN
            # Predict Q values from state
            Q_values = self.sess.run(self.network.output, feed_dict={self.network.inputs_:state.reshape((1, *state.shape))})

            # Take the biggest Q value
            return one_hot_encode_action(np.argmax(Q_values), self.env.action_space.n), explore_probability

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        for episode in range(self.total_episodes):
            step = 0 
            show_episode = False
            done = False
            episode_rewards = []
            state = self.env.reset()
            state, stacked_frames = stack_frames(deque(), state, True, stack_size=self.stack_size)

            if episode % (self.total_episodes / self.debug_rate ) <= 1:
                save_path = self.saver.save(self.sess, self.frozen_model_dir)
                if self.debug:
                    show_episode = True
                    print("Model Saved...\nShowing next episode ({}/{})...\n".format(episode, self.total_episodes))


            while not done:
                # If debug enabled and at a far enough checkpoint show episode
                if self.debug and show_episode:
                    self.env.render()

                self.decay_step += 1

                action, explore_probability = self.predict_action(state)
                next_state, reward, done, _ = self.env.step(one_hot_decode_action(action))

                episode_rewards.append(reward)

                if done:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=self.stack_size)
                    # Calc total reward of episode
                    total_reward = np.sum(episode_rewards)

                    if self.debug:
                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              # 'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=self.stack_size)
                self.memory.add((state, action, reward, next_state, done))

                # State + 1 is now the current state
                state = next_state

            ## Learning part
            # Obtain random mini-batches from memory to train on
            batch = self.memory.sample(self.batch_size)
            states_minibatch = np.array([sample[0] for sample in batch])
            actions_minibatch = np.array([sample[1] for sample in batch])
            rewards_minibatch = np.array([sample[2] for sample in batch])
            next_states_minibatch = np.array([sample[3] for sample in batch])
            done_minibatch = np.array([sample[4] for sample in batch])
            target_Qs_batch = []

            # Get Qs for next_states
            Qs_next_state = self.sess.run(self.network.output, feed_dict={self.network.inputs_:next_states_minibatch})

            # Set Qtarget to reward if episode ends at s+1 otherwise Qtarget = reward * gamma*maxQ(s',a')
            for i in range(0, len(batch)):
                terminal = done_minibatch[i]

                if terminal:
                    target_Qs_batch.append(rewards_minibatch[i])
                else:
                    target = rewards_minibatch[i] + self.gamma * np.max(Qs_next_state[i])
                    target_Qs_batch.append(target)

            targets_minibatch = np.array(target_Qs_batch)

        
            feed = {self.network.inputs_:states_minibatch,
                           self.network.target_Q: targets_minibatch,
                           self.network.actions_: actions_minibatch}

            loss, _ = self.sess.run([self.network.loss, self.network.optimizer],
                feed_dict=feed)

            # TODO combine with above
            summary = self.sess.run(self.write_op,feed_dict=feed)
            self.writer.add_summary(summary, episode)
            self.writer.flush()

    def play(self, num_episodes=1):
        self.saver.restore(self.sess, self.frozen_model_dir)
        for i in range(num_episodes):
            total_score = 0
            done = False
            state = self.env.reset()
            state, stacked_frames = stack_frames(deque(), state, True, stack_size=self.stack_size)
            while not done:
                # Get predicted q-values from trained network (if you trained it)
                Q_values = self.sess.run(self.network.output, feed_dict={self.network.inputs_:state.reshape((1, *state.shape))})
                # Always act greedily (take argmax of q-values)
                action = np.argmax(Q_values)
                # Run with that action 
                state, reward, done, _ = self.env.step(action)
                
                # Add to current total_rewards
                total_score += reward
                # Convert back into stacked form for next prediction 
                state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=self.stack_size)

                self.env.render()

            if self.debug:
                print("Finished episode {} with a total score of {}".format(i, total_score))

def preprocess_image(frame, resize_shape=(120,120)):
    return transform.resize(color.rgb2gray(frame), [120,120], anti_aliasing=True)

def stack_frames(stacked_frames, state, is_new_episode, stack_size=4, preprocess_if_image=True):
    state = preprocess_image(state) if preprocess_if_image and len(state.shape) > 2 else state
    if is_new_episode:
        stacked_frames = deque([np.zeros(state.shape, dtype=np.int) for i in range(stack_size)], maxlen=stack_size)

        # Since we're in a new episode coppy the same frame stack_size times
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)

    return np.stack(stacked_frames, axis=len(state.shape)), stacked_frames

def one_hot_encode_action(action, action_space):
    return np.identity(action_space)[action]

def one_hot_decode_action(action):
    return np.argmax(action)

class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork', dense_architecture=[120, 40],
                 dropout=True, activation=tf.nn.relu, dropout_rate=0.25, conv_architecture=[(32,8,4), (64,4,2), (128,4,2)],
                 batch_norm=True):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.image_data = True if len(state_size) > 2 else False

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Create placeholders
            # Where *state_size just unfolds the state_size array e.g.: if state_size was [5,5] then we'd write [None, 5, 5]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None,action_size], name='actions')

            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            # Initialize with Flattening layer (flatten stack, probably not best thing to do but w.e) if not image
            
            convolution = self.inputs_

            if self.image_data:
                conv_layer = 1

                for num_filters, kernel_size, strides in conv_architecture:
                    convolution = tf.layers.conv2d(inputs=convolution, filters=num_filters, kernel_size=[kernel_size, kernel_size],
                        strides=[strides,strides], padding="VALID", kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        name="convolution_{}".format(conv_layer), activation=activation)
                    if batch_norm:
                        convolution = tf.layers.batch_normalization(convolution, training=True, name="batch_norm_{}".format(conv_layer)) 

                    conv_layer += 1

            fc_i_layer = tf.layers.flatten(convolution)

            for i, layer_units in enumerate(dense_architecture):
                fc_i_layer = tf.layers.dense(inputs=fc_i_layer, units=layer_units, activation = activation, name='full_connected_{}'.format(i))
                if dropout:
                    fc_i_layer = tf.layers.dropout(inputs=fc_i_layer, rate=dropout_rate)

            self.output = tf.layers.dense(inputs=fc_i_layer, units=action_size)

            # Q is predicted Q
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the square diff between predicted Q and target Q
            self.loss = tf.reduce_mean(tf.square(self.Q - self.target_Q))

            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, env, max_size, pretrain_length, stack_size):
        self.buffer = deque(maxlen=max_size)
        self.stack_size = stack_size

        self._prepopulate(env, pretrain_length)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        return [self.buffer[i] for i in index]

    def _prepopulate(self, env, pretrain_length):
        for i in range(pretrain_length):
            if i == 0:
                state = env.reset()
                state, stacked_frames = stack_frames(None, state, True, stack_size=self.stack_size)

            # Random action
            action = one_hot_encode_action(env.action_space.sample(), env.action_space.n)

            next_state, reward, done, _  = env.step(one_hot_decode_action(action))
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False, stack_size=self.stack_size)

            if done:

                self.add((state, action, reward, next_state, done))

                # Reset
                state = env.reset()

                state, stacked_frames = stack_frames(stacked_frames, state, True, stack_size=self.stack_size)

            else:
                self.add((state, action, reward, next_state, done))
                state = next_state



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Slightly more complex deep q-learning algorithm implementation")

    parser.add_argument('--env_name', type=str, default='CartPole-v0')
    parser.add_argument('--learning_rate', type=float, default=0.0002)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--total_episodes', type=int, default=2000)
    parser.add_argument('--explore_start', type=float, default=1.0)
    parser.add_argument('--explore_stop', type=float, default=0.01)
    parser.add_argument('--decay_rate', type=float, default=0.0001)
    parser.add_argument('--memory_size', type=int, default=1000000)
    parser.add_argument('--pretrain_length', type=int, default=2000)
    parser.add_argument('--stack_size', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=1)
    parser.add_argument('--debug_rate', type=int, default=6)


    parser.add_argument('--dropout', action='store_true')
    parser.add_argument('--dropout_rate', type=float, default=0.25)

    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--play', action='store_true')

    args = vars(parser.parse_args())

    play = args.pop('play')

    num_episodes = args.pop('num_episodes')
    print(args)
    agent = Agent(**args)

    if play:
        agent.play(num_episodes)
    else:
        agent.train()
        
    

    
