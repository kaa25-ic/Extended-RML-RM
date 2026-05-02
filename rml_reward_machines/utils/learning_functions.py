import yaml
import pickle
import random
from envs.letterenv import LetterEnv
import pandas as pd
import numpy as np

def learning_episode_office(rewards, env, q_table, actions, alpha, gamma, epsilon, total_steps,n, reward_if_correct=[110,112]):
    succesful_policy = False
    env.env.set_n(n)
    state , _ = env.reset()

    if isinstance(state['monitor'], int):
        state_tuple = (state['position'], (state['monitor']))
    else:
        state_tuple = (state['position'], tuple(state['monitor']))

    done = False
    total_reward = 0
    while not done:
        valid_actions = [a for a in actions if (state_tuple[0],state_tuple[1],a) not in env.env.env.env.env.env.forbidden_transitions]
        if state_tuple not in q_table:         # Add state to the q table
            q_table[state_tuple] = {action: 0 for action in valid_actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(valid_actions)
        else:
            max_value = max(q_table[state_tuple].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in valid_actions if q_table[state_tuple][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, _, __ = env.step(action)#
        if isinstance(next_state['monitor'], int):
            next_state_tuple = (next_state['position'], (next_state['monitor']))
        else:
            next_state_tuple = (next_state['position'], tuple(next_state['monitor']))

        if next_state_tuple not in q_table:    # Add next state to the q table
            q_table[next_state_tuple] = {action: 0 for action in valid_actions}
            #reward += 2

        # Q-learning update
        old_value = q_table[state_tuple][action]
        next_max = max(q_table[next_state_tuple].values())
        q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state_tuple = next_state_tuple
        total_steps += 1
        total_reward += reward

    # Decay epsilon
    epsilon *= 0.9999
        
    rewards.append(reward)    # Only checking the last reward as it contains information on whether the episode is a success

    if rewards.count(110) > 0:
        # Succesful policy true when deterministic policy is succesful, false otherwise
        succesful_policy = eval_office_world(env, q_table, actions, n)
    
    return rewards, succesful_policy, q_table, state, epsilon, total_steps

def learning_episode_letter(rewards, env, q_table, actions, alpha, gamma, epsilon, total_steps,n,reward_if_correct=[110,112]):
    succesful_policy = False
    env.env.set_n(n)
    state , _ = env.reset()

    if isinstance(state['monitor'], int):
        state_tuple = (state['position'], (state['monitor']))
    else:
        state_tuple = (state['position'], tuple(state['monitor']))

    done = False
    total_reward = 0
    while not done:
        if state_tuple not in q_table:         # Add state to the q table
            q_table[state_tuple] = {action: 0 for action in actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            max_value = max(q_table[state_tuple].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in actions if q_table[state_tuple][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, _, __ = env.step(action)#
        if isinstance(next_state['monitor'], int):
            next_state_tuple = (next_state['position'], (next_state['monitor']))
        else:
            next_state_tuple = (next_state['position'], tuple(next_state['monitor']))

        if next_state_tuple not in q_table:    # Add next state to the q table
            q_table[next_state_tuple] = {action: 0 for action in actions}
            reward += 2
        # Q-learning update
        old_value = q_table[state_tuple][action]
        next_max = max(q_table[next_state_tuple].values())
        q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state_tuple = next_state_tuple
        total_steps += 1
        total_reward += reward

    # Decay epsilon
    epsilon *= 0.99
    rewards.append(reward)    # Only checking the last reward as it contains information on whether the episode is a success
    if len(rewards) >= 20:
        average_reward = sum(rewards[-20:])/20

        if average_reward in reward_if_correct:
            succesful_policy = True
    
    return rewards, succesful_policy, q_table, state, epsilon, total_steps

def learning_episode_letter_no_exploration(rewards, env, q_table, actions, alpha, gamma, epsilon, total_steps,n,reward_if_correct=[110,112]):
    succesful_policy = False
    env.env.set_n(n)
    state , _ = env.reset()

    if isinstance(state['monitor'], int):
        state_tuple = (state['position'], (state['monitor']))
    else:
        state_tuple = (state['position'], tuple(state['monitor']))

    done = False
    total_reward = 0
    while not done:
        if state_tuple not in q_table:         # Add state to the q table
            q_table[state_tuple] = {action: 0 for action in actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            max_value = max(q_table[state_tuple].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in actions if q_table[state_tuple][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, _, __ = env.step(action)#
        if isinstance(next_state['monitor'], int):
            next_state_tuple = (next_state['position'], (next_state['monitor']))
        else:
            next_state_tuple = (next_state['position'], tuple(next_state['monitor']))

        if next_state_tuple not in q_table:    # Add next state to the q table
            q_table[next_state_tuple] = {action: 0 for action in actions}

        # Q-learning update
        old_value = q_table[state_tuple][action]
        next_max = max(q_table[next_state_tuple].values())
        q_table[state_tuple][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state_tuple = next_state_tuple
        total_steps += 1
        total_reward += reward

    # Decay epsilon
    epsilon *= 0.99
    rewards.append(reward)    # Only checking the last reward as it contains information on whether the episode is a success
    if len(rewards) >= 20:
        average_reward = sum(rewards[-20:])/20

        if average_reward in reward_if_correct:
            succesful_policy = True
    
    return rewards, succesful_policy, q_table, state, epsilon, total_steps

def eval_office_world(env, q_table, actions, n):
    env.env.set_n(n)
    state, _ = env.reset()
    max_steps = 500
    steps = 0

    if isinstance(state['monitor'], int):
        state_tuple = (state['position'], (state['monitor']))
    else:
        state_tuple = (state['position'], tuple(state['monitor']))

    done = False
    total_reward = 0

    while not done and steps < max_steps:
        if state_tuple not in q_table:
            return False

        valid_actions = [a for a in actions if (state_tuple[0], state_tuple[1], a) not in env.env.env.env.env.env.forbidden_transitions]
        if not valid_actions:   # Failure if no available action
            return False
        
        # Select the best action 
        max_value = max(q_table[state_tuple].values())
        best_actions = [a for a in valid_actions if q_table[state_tuple][a] == max_value]
        action = random.choice(best_actions)

        # Take action
        next_state, reward, done, _, __ = env.step(action)

        if isinstance(next_state['monitor'], int):
            next_state_tuple = (next_state['position'], (next_state['monitor']))
        else:
            next_state_tuple = (next_state['position'], tuple(next_state['monitor']))

        state_tuple = next_state_tuple
        total_reward += reward
        steps += 1
 
    # Check if the total_reward indicates a successful episode
    return reward in [110, 112]



def evaluation_episode_encoding(env, q_table, actions, 
                        n, total_episodes, total_steps, result_table, reward_if_correct, max_steps = 500):
    """
    Code is used to evaluate when the environment, how long it takes to a succesful policy.
    Needs as input total training episodes and steps (as well as other relevant items)
    """
    
    succesful_policy = False
    env.env.set_n(n)
    state, _ = env.reset()

    if isinstance(state['monitor'], int):
        state_tuple = (state['position'], (state['monitor']))
    else:
        state_tuple = (state['position'], tuple(state['monitor']))    
    
    done = False
    total_reward = 0
    n_steps = 0

    while not done:
        if state_tuple in q_table:
            max_value = max(q_table[state_tuple].values())
            best_actions = [a for a in actions if q_table[state_tuple][a] == max_value]
            action = random.choice(best_actions)
            # Take action
            next_state, reward, done, _, __ = env.step(action)
            total_reward += reward
            if isinstance(state['monitor'], int):
                state_tuple = (next_state['position'], (next_state['monitor']))
            else:
                state_tuple = (next_state['position'], tuple(next_state['monitor'])) 
                 
            n_steps += 1
            if n_steps > max_steps:
                done = True
        else:
            #action = random.choice(valid_actions)
            done = True
    if reward == reward_if_correct:  # Using the reward given for finishing as the ending condition
        print('n val - ', n)
        new_row = pd.DataFrame([{'n value': n, 'episodes': total_episodes, 'steps': total_steps}])
        result_table = pd.concat([result_table, new_row])
        succesful_policy = True

        
    return succesful_policy, result_table


def learning_episode_letter_original(rewards, env, q_table, actions, alpha, gamma, epsilon, n,decay):
    env.env.set_n(n)
    state = env.reset()
    total_steps = 0
    done = False
    truncated = False
    total_reward = 0
    while not done and not truncated:
        if state not in q_table:         # Add state to the q table
            q_table[state] = {action: 0 for action in actions}
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            max_value = max(q_table[state].values())
            # Find all actions that have this maximum Q-value among valid actions
            best_actions = [a for a in actions if q_table[state][a] == max_value]
            # Randomly choose among the best actions
            action = random.choice(best_actions)
        # Take action
        next_state, reward, done, truncated, __ = env.step(action)

        if next_state not in q_table:    # Add next state to the q table
            q_table[next_state] = {action: 0 for action in actions}

        # Q-learning update
        old_value = q_table[state][action]
        next_max = max(q_table[next_state].values())
        q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state
        total_steps += 1
        total_reward += reward

        if total_steps > 200:
            print(total_steps)

    # Decay epsilon
    epsilon *= decay
    rewards.append(reward)    # Only checking the last reward as it contains information on whether the episode is a success

    
    return rewards, q_table, epsilon