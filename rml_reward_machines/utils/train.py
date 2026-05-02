import numpy as np

from .eval import eval_agent

import time 
import copy
from tqdm import tqdm
from environments.context_free_conditional import create_context_free_env_conditional_labelled
from environments.context_free_conditional_multiplicative import create_context_free_env_conditional_multiplicative_labelled
from environments.context_free_conditional_additive import create_context_free_env_conditional_additive_labelled

def train_repeat(
    agent,
    env,
    n_episodes,
    n_repeats=1,
):
    rewards = []

    for _ in range(n_repeats):
        agent.reset_training()
        reward = train(
            agent,
            env,
            n_episodes,
        )
        rewards.append(reward)

    rewards = np.array(rewards)
    return np.mean(rewards, axis=0)


def train_till_conv_repeat(
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_2(
            agent,
            env,
            max_samples // 100,
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)

def train_till_conv_repeat_letter(
    actions,
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_4(
            actions,
            agent,
            env,
            max_samples // 200,
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)

def train_till_conv_repeat_letter_no_counterfactual(
    actions,
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_5(
            actions,
            agent,
            env,
            max_samples // 200,
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)

def train_till_conv_repeat_office(
    actions,
    agent,
    env,
    max_samples,
    n_repeats=1,
):
    sample_counts = []

    for _ in range(n_repeats):
        agent.reset_training()
        sample_count = train_conv_3(
            agent,
            env,
            max_samples // 100,
            actions
        )
        sample_counts.append(sample_count)
        # print(sample_count)

    return np.mean(sample_counts)


def train(
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []

    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(100):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)

            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            # if reward == 1:
            #     print("HERE")

            # if terminated:
            #     break
            if agent.terminated():
                # print(obs, next_obs, last_u, agent.u)
                break

        agent.decay_epsilon()
        rewards.append(reward)

        # if episode % 1000 == 0:
        #     pbar.set_description(
        #         f"Episode {episode} | Reward {np.mean(rewards[-1000:]):.2f}"
        #     )
    return np.array(rewards)


def train_till_conv(
    agent,
    env,
    max_samples
):
    sample_counter = 0
    done = False

    while not done:
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(200):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)

            sample_counter += 1

            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            if agent.terminated():
                break
            if sample_counter == max_samples:
                break

        agent.decay_epsilon()
        done = eval_agent(agent, env) == 1 or sample_counter == max_samples
    return sample_counter


def train_conv_2(
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(100):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break

        agent.decay_epsilon()
        rewards.append(reward)

        if episode >= 20:
            ave = np.mean(rewards[-20:])
            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                break
    return sample_count

def train_conv_3(
    agent,
    env,
    n_episodes,
    actions
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        for t in range(500):
            action = agent.get_action(obs,env,actions)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)

            obs = next_obs
            
            sample_count += 1

            if agent.terminated() or terminated:
                break

        agent.decay_epsilon()
        rewards.append(reward)

        if rewards.count(1) > 0:
            # Succesful policy true when deterministic policy is succesful, false otherwise
            succesful_policy = eval_office_world(env, agent)
            

            if succesful_policy:
                break

    return sample_count

def eval_office_world(env,agent):
    agent.reset()
    obs, _ = env.reset()
    obs = env.observation(obs)
    success = False
    steps = 0
    for t in range(500):
        action, flag = agent.get_evaluation_action(obs)
        if flag:   # Breaks if the agent has 0 in q table for all possible next actions
            if steps == 0:
                reward = 0
            break

        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.observation(next_obs)
        agent.evaluation_update(next_obs)
        obs = next_obs
        
        steps += 1
        if agent.terminated() or terminated:
            break
    if reward == 1:
        success = True
    return success



def train_conv_4(
    actions,
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    length = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)
        symbols = []
        for t in range(200):
            action = agent.get_action(obs,env,actions)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break


        agent.decay_epsilon()
        rewards.append(reward)

        if episode >= 20:
            ave = np.mean(rewards[-20:])

            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                break
        prev_count = copy.deepcopy(sample_count)
    return sample_count

def train_conv_5(
    actions,
    agent,
    env,
    n_episodes,
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    length = 0
    for episode in range(n_episodes):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)
        symbols = []
        for t in range(200):
            action = agent.get_action(obs)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)
            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break


        agent.decay_epsilon()
        rewards.append(reward)

        if episode >= 20:
            ave = np.mean(rewards[-20:])

            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                break
        prev_count = copy.deepcopy(sample_count)
    return sample_count

def train_deep_avg_reward(
    agent,
    env,
    n_episodes,
    batch_size,
    warmup_steps=100,
    update_every=4,
    update_target_every=500,
    max_steps_per_episode=100
):
    rewards_per_episode = []
    objects_collected_per_episode = []
    total_steps = 0

    for episode in tqdm(range(n_episodes)):
        agent.reset()
        obs, _ = env.reset()
        obs = env.observation(obs)

        total_reward = 0
        objects_collected = 0
        for t in range(max_steps_per_episode):
            # Get action from agent
            action = agent.get_action(obs)

            # Take action in environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = env.observation(next_obs)
            done = terminated or truncated

            # Update agent's machine state and store experience in buffer
            r = agent.step(obs, action, next_obs, done)   # Actual reward recieved by agent

            # Update Q-network (call periodically or every step)
            if total_steps > warmup_steps and total_steps % update_every == 0:
                agent.update(batch_size)  # Update the Q-network

            # Optionally update target network periodically
            if total_steps % update_target_every == 0:
                agent.update_target_network()

            # Move to next state
            obs = next_obs

            total_reward += r
            total_steps += 1
            if r == 1:
                objects_collected += 1

            if agent.terminated() or done:
                break

        # Decay epsilon
        agent.decay_epsilon()

        # Save total reward and objects collected for this episode
        rewards_per_episode.append(total_reward)
        objects_collected_per_episode.append(objects_collected)
        # Print average reward every 100 episodes
        if (episode + 1) % 100 == 0:
            avg_reward = sum(rewards_per_episode[-100:]) / 100
            print(f"Episode {episode + 1}, Average Reward (Last 100): {avg_reward}")
            avg_collected = sum(objects_collected_per_episode[-100:]) / 100
            print(f"Episode {episode + 1}, Objects Collected (Last 100): {avg_collected}")
            print(f'Epsilon - {agent.epsilon}')


    return rewards_per_episode, objects_collected_per_episode


def train_letter_conditional(
    actions,
    agent,
    n_episodes,
    N
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    sample_count = 0
    agent.reset_training()

    for episode in range(n_episodes): 
        agent.reset()
        env = create_context_free_env_conditional_labelled(N)
        obs, _ = env.reset()
        obs = env.observation(obs)
        for t in range(100):
            action = agent.get_action(obs,env,actions)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)

            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break


        agent.decay_epsilon()
        rewards.append(reward)

        if episode >= 20:
            ave = np.mean(rewards[-20:])

            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                break
    return sample_count

def train_letter_conditional_multiplicative(
    actions,
    agent,
    n_episodes,
    N
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    counters = []
    sample_count = 0
    agent.reset_training()

    for episode in range(n_episodes): 
        agent.reset()
        env = create_context_free_env_conditional_multiplicative_labelled(N)
        obs, _ = env.reset()
        obs = env.observation(obs)
        for t in range(100):
            action = agent.get_action(obs,env,actions)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)

            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break


        agent.decay_epsilon()
        rewards.append(reward)
        counters.append(agent.counters)

        if episode >= 20:
            ave = np.mean(rewards[-20:])

            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                #print(counters[-20:])
                break
    return sample_count

def train_letter_conditional_additive(
    actions,
    agent,
    n_episodes,
    N
):
    # pbar = tqdm(range(n_episodes))
    rewards = []
    counters = []
    sample_count = 0
    agent.reset_training()

    for episode in range(n_episodes): 
        agent.reset()
        env = create_context_free_env_conditional_additive_labelled(N)
        obs, _ = env.reset()
        obs = env.observation(obs)
        for t in range(100):
            action = agent.get_action(obs,env,actions)
            next_obs, reward, terminated, _, _ = env.step(action)
            next_obs = env.observation(next_obs)
            agent.update(obs, action, next_obs, reward, terminated)

            obs = next_obs

            sample_count += 1

            if agent.terminated():
                break


        agent.decay_epsilon()
        rewards.append(reward)
        counters.append(agent.counters)

        if episode >= 20:
            ave = np.mean(rewards[-20:])

            # pbar.set_description(
            #     f"Episode {episode} | Reward {np.mean(rewards[-100:]):.2f}"
            # )
            if ave == 1:
                #print(counters[-20:])
                break
    return sample_count