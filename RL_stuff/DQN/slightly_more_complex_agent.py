import tensorflow as tf
import numpy as np

import gym

import random
import time
from collections import deque

import warnings

warnings.filterwarnings('ignore')

class Agent:
    def __init__(self, env, total_episodes=500, max_steps=100, state_shape=[4,4], batch_size=64, explore_start=1.0, explore_stop=0.01, decay_rate=0.0001, action_size=2, debug=False, learning_rate = 0.0002, memory_size = 1000000, gamma=0.95):
        self.env = env
        self.network = DQNetwork(state_shape, action_size, learning_rate)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.total_episodes = total_episodes
        self.memory = Memory(env, memory_size, batch_size)
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.debug=debug
        self.explore_start = explore_start
        self.explore_stop = explore_stop
        self.decay_rate = decay_rate
        self.decay_step = 0
        self.gamma = gamma 
         # Setup tensorboard 
        self.writer = tf.summary.FileWriter("scratch/dqn")

        # Loss
        tf.summary.scalar("DQN_Loss", self.network.loss)
        self.write_op = tf.summary.merge_all()

    def predict_action(self, state):
        exp_exp_tradeoff = np.random.randn()

        explore_probability = self.explore_stop + (self.explore_start - self.explore_start) * np.exp(-self.decay_rate*self.decay_step)

        if (explore_probability > exp_exp_tradeoff):
            # Take random action
            return self.env.action_space.sample(), explore_probability

        else:
            # Predict action from DQN
            # Predict Q values from state
            Q_values = self.sess.run(self.network.output, feed_dict={self.network.inputs_:state.reshape((1, *state.shape))})

            # Take the biggest Q value
            return np.argmax(Q_values), explore_probability

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        # Init decay step for epsilon greedy algo
        decay_step = 0

        for episode in range(self.total_episodes):
            step = 0 
            episode_rewards = []
            state = self.env.reset()
            state, stacked_frames = stack_frames(deque(), state, True)

            if episode % 5 == 0:
                save_path = self.saver.save(self.sess, "./models/model.ckpt")
                print("Model Saved...\n")

            while step < self.max_steps:
                step+=1

                decay_step += 1

                action, explore_probability = self.predict_action(state)
                next_state, reward, done, _ = self.env.step(action)

                episode_rewards.append(reward)

                if done:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
                    # Stop the episode by setting step to max_steps
                    step = self.max_steps
                    # Calc total reward of episode
                    total_reward = np.sum(episode_rewards)

                    if self.debug:
                        print('Episode: {}'.format(episode),
                              'Total reward: {}'.format(total_reward),
                              # 'Training loss: {:.4f}'.format(loss),
                              'Explore P: {:.4f}'.format(explore_probability))
                else:
                    next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

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

            # Get Qs for next_state
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

            print(actions_minibatch)
            feed = {self.network.inputs_:states_minibatch,
                           self.network.target_Q: targets_minibatch,
                           self.network.actions_: actions_minibatch}

            loss, _ = self.sess.run([self.network.loss, self.network.optimizer],
                feed_dict=feed)

            # TODO combine with above
            summary = self.sess.run(self.write_op,feed_dict=feed)
            self.writer.add_summary(summary, episode)
            self.writer.flush()


def stack_frames(stacked_frames, state, is_new_episode, stack_size=4, max_len=4):
    if is_new_episode:
        stacked_frames = deque([np.zeros(state.shape, dtype=np.int) for i in range(stack_size)], maxlen=max_len)

        # Since we're in a new episode coppy the same frame stack_size times
        for _ in range(stack_size):
            stacked_frames.append(state)
    else:
        stacked_frames.append(state)

    return np.stack(stacked_frames), stacked_frames


class DQNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork', hidden_layer_size = 120):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # Create placeholders
            # Where *state_size just unfolds the state_size array e.g.: if state_size was [5,5] then we'd write [None, 5, 5]
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name='inputs')
            self.actions_ = tf.placeholder(tf.float32, [None,], name='actions')

            self.target_Q = tf.placeholder(tf.float32, [None], name='target')

            self.flatten = tf.layers.flatten(self.inputs_)
            self.fc = tf.layers.dense(inputs=self.flatten, units=hidden_layer_size, activation=tf.nn.elu, name='fc1')
            self.output = tf.layers.dense(inputs=self.fc, units=action_size)

            # Q is predicted Q
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is the square diff between predicted Q and target Q
            self.loss = tf.reduce_mean(tf.square(self.Q - self.target_Q))

            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)


class Memory():
    def __init__(self, env, max_size, pretrain_length):
        self.buffer = deque(maxlen=max_size)
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
                state, stacked_frames = stack_frames(None, state, True)

            # Random action
            action = env.action_space.sample()

            next_state, reward, done, _  = env.step(action)
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)

            if done:

                self.add((state, action, reward, next_state, done))

                # Reset
                state = env.reset()

                state, stacked_frames = stack_frames(stacked_frames, state, True)

            else:
                self.add((state, action, reward, next_state, done))
                state = next_state



def __main__(stack_size=4, env_name='CartPole-v1', learning_rate=0.0002, gamma=0.95, batch_size=64, total_episodes=500, memory_size=1000000):
    
    pretrain_length = batch_size



