import argparse
import gym
import numpy as np
import random
import tensorflow as tf
import time

def __main__(env_name='FrozenLake-v0', learning_rate=0.1, gamma=0.99, epsilon=0.1, num_episodes=5000,
             debug=False, debug_scale=500):
    
    env = gym.make(env_name)

    # Reset tensorflow graph
    tf.reset_default_graph()

    # Feed-forward part of the network
    env_size = env.observation_space.n
    action_size = env.action_space.n
    inputs1 = tf.placeholder(shape=[1,env_size], dtype=tf.float32)
    W = tf.Variable(tf.random_uniform([env_size, action_size], 0, 0.01))
    Qout = tf.matmul(inputs1, W)
    predict = tf.argmax(Qout,1)

    # Define loss function (sum of square difference between target and prediction Q values)
    nextQ = tf.placeholder(shape=[1,action_size], dtype=tf.float32)
    loss = tf.reduce_sum(tf.square(nextQ - Qout))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    updateModel = optimizer.minimize(loss)

    # Train
    init = tf.initialize_all_variables()

    # Lists for total rewards and steps to get to final state in that episode
    total_rewards = []
    steps_to_complete = []
    with tf.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            show_episode = (debug and i % debug_scale == 0)
            if show_episode:
                print("Showing epsiode {}/{} ...\n".format(i, num_episodes))
                time.sleep(1)
            state = env.reset()
            cumulative_reward = 0
            done = False
            current_step = 0
            # DQN
            while not done:

                if show_episode:
                    env.render()
                    time.sleep(0.5)

                current_step += 1
                # Predict next action using NN rather that querying Q-table
                one_hot_encoded_state = np.identity(env_size)[state:state+1]
                action, q_values = sess.run([predict, Qout], feed_dict={inputs1:one_hot_encoded_state})

                # Epsilon greedy (bigger epsilon more random)
                if np.random.rand(1) < epsilon:
                    # Replace chosen action with random action
                    action[0] = env.action_space.sample()

                # Take action (either chosen or random)
                new_state, reward, done, _ = env.step(action[0])

                one_hot_encoded_new_state = np.identity(env_size)[new_state:new_state+1]
                Q1 = sess.run(Qout, feed_dict={inputs1:one_hot_encoded_new_state})

                maxQ1 = np.max(Q1)
                targetQ = q_values
                targetQ[0,action[0]] = reward + gamma*maxQ1

                _,W1 = sess.run([updateModel,W], feed_dict={inputs1:one_hot_encoded_state, nextQ:targetQ})

                # For logging
                cumulative_reward += reward
                # Move to next state
                state = new_state

                if done:
                    epsilon = 1.0/((i/50) + 10)
                    break

            steps_to_complete.append(current_step)
            total_rewards.append(cumulative_reward)

            if show_episode:
                print("Completed with cumulative reward of {}...\n".format(cumulative_reward))
                time.sleep(1)

    return total_rewards
    print("Percent of success: {}...\n".format(sum(total_rewards)/num_episodes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Simple deep q-learning algorithm implementation")
    parser.add_argument('--env_name', type=str, default='FrozenLake-v0')
    parser.add_argument('--learning_rate', type=float, default=0.80)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--epsilon', type=float, default=0.1)
    parser.add_argument('--num_episodes', type=int, default=2000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_scale', type=int, default=500)
    args = parser.parse_args()

    total_rewards = __main__(env_name=args.env_name, learning_rate=args.learning_rate, gamma=args.gamma,
             epsilon=args.epsilon, num_episodes=args.num_episodes, debug=args.debug, debug_scale=args.debug_scale)
    
    print("Percent of success : {}...\n".format(sum(total_rewards)/args.num_episodes))

