import argparse
import gym
import numpy as np
import time

def __main__(env_name='FrozenLake-v0', learning_rate = 0.8, gamma=0.95, num_episodes=4000, debug=False, debug_scale=1000):
    # Make frozen lake environment
    env = gym.make(env_name)

    # Initialize the Q table to be the size of the observation space x action space (all possible state action pair)
    Q = np.zeros([env.observation_space.n, env.action_space.n])

    # For list of previous rewards
    past_rewards = []
    for i in range(num_episodes):
        show_episode = False
        if debug and i % debug_scale==0:
            show_episode = True
            print("\nEpisode {}/{}...\n".format(i, num_episodes))
            time.sleep(1)
        # Reset env and get first new observations
        state = env.reset()
        total_rewards = 0
        done = False
        steps = 0
        # Q learning algorithm
        while not done:

            # For visualization of algorithm
            if show_episode:
                env.render()
                time.sleep(0.25)

            steps += 1

            # Greedily choose (with noise) action picking from Q-table
            # Noise is scaled so later steps are more confidently greedy
            noise_scaling_factor = 1.0/(i+1)
            noise = noise_scaling_factor * np.random.randn(1,env.action_space.n)
            # Take argmax of all possible Q values for given state plus some randomness added to each value
            possible_actions_given_state = Q[state,:]
            action = np.argmax(possible_actions_given_state + noise)

            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)

            # Update Q-table with new state and reward using Bellman equation
            update = reward + gamma * np.max(Q[new_state,:]) - Q[state, action]
            Q[state, action] = Q[state, action] + learning_rate * update

            # Update total_rewards and state with new reward and state
            total_rewards += reward
            state = new_state

            if done == True:
                break
        if show_episode:
            print('Finished with a total rewards of {}...\n'.format(total_rewards))
            time.sleep(1)
        past_rewards.append(total_rewards)
    return past_rewards, Q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Basic Q-Table algorithm implementation")
    parser.add_argument('--env_name', type=str, default='FrozenLake-v0')
    parser.add_argument('--learning_rate', type=float, default=0.80)
    parser.add_argument('--gamma', type=float, default=0.95)
    parser.add_argument('--num_episodes', type=int, default=2000)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_scale', type=int, default=500)
    args = parser.parse_args()

    rewards_over_time, Q = __main__(env_name=args.env_name, learning_rate=args.learning_rate, gamma=args.gamma,
        num_episodes=args.num_episodes, debug=args.debug, debug_scale=args.debug_scale)


    print("Score over time: {}...\n".format(sum(rewards_over_time)/args.num_episodes))
    print("Final Q-Table values : \n {}".format(Q))

