import random
from tqdm import tqdm
from gymnasium.envs.registration import register
from utils.encoding_functions import create_encoding, generate_events_and_index


def learning_episode_new_vector(environment, config_path, Actions, total_episodes, hyperparameters):
    states_for_encoding = {0: '@(eps*(star(not_abcd:eps)*app(,[0])),[=gen([n],(a_match:eps)*(star(not_abcd:eps)*(app(,[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1])))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        1: '@(star(not_abcd:eps)*(app(gen([n],),[0+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[0+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        2: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[0+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[0+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        3: '@(star(not_abcd:eps)*(app(gen([n],),[1+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[1+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        4: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[1+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[1+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        5: '@(star(not_abcd:eps)*(app(gen([n],),[2+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[2+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        6: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[2+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[2+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        7: '@(star(not_abcd:eps)*(app(gen([n],),[3+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[3+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        8: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[3+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[3+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        9: 'false_verdict', 
        10: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[1]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        11: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        12: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[2]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        13: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[2]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        14: '@(app(gen([n],),[2]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        15: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[2-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        16: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[3]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        17: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[3]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        18: '@(app(gen([n],),[1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        19: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[1-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        20: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[4]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        21: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[4]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        22: '@(star(not_abcd:eps)*(app(gen([n],),[4+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[4+1])),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        23: '@(eps*(star(not_abcd:eps)*(app(gen([n],),[4+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[4+1]))),[=(a_match:eps)*(star(not_abcd:eps)*(app(gen([n],),[var(n)+1])\\/app(gen([n],(b_match:eps)*app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[var(n)])),[var(n)+1]))),=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        24: '@(app(gen([n],star(not_abcd:eps)*((c_match:eps)*app(,[var(n)]))),[5]),[=gen([n],guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(,[var(n)-1])),1))])', 
        25: '@(eps*(star(not_abcd:eps)*((c_match:eps)*app(gen([n],),[5]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        26: '@(app(gen([n],),[4]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        27: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[4-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        28: '@(app(gen([n],),[3]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        29: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[3-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        30: '@(app(gen([n],),[4-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        31: '@(app(gen([n],),[3-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        32: '@(app(gen([n],),[2-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        33: '@(app(gen([n],),[1-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        34: '1', 
        35: '@(app(gen([n],),[5]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        36: '@(eps*(star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[5-1]))),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])', 
        37: '@(app(gen([n],),[5-1]),[=guarded(var(n)>0,star(not_abcd:eps)*((d_match:eps)*app(gen([n],),[var(n)-1])),1)])'}
    
    unique_events, event_index = generate_events_and_index(states_for_encoding)
    initial_encoding = create_encoding(states_for_encoding[0],event_index)

    register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:RML_LetterEnv_5',
    max_episode_steps=200
    )
    env = environment(event_index, initial_encoding, config_path)

    env.env.set_n(1)
    
    
    q_table = {}
    alpha, gamma, epsilon, eps_decay = hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    episode_rewards = []
    successes = []
    for episode in tqdm(range(total_episodes)):
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
            total_reward += reward

        # Decay epsilon
        epsilon *= eps_decay

        episode_rewards.append(total_reward)    # Only checking the last reward as it contains information on whether the episode is a success
        if reward >= 20:
            successes.append(episode)

    
    return episode_rewards, successes

def learning_episode_new_simple(environment, config_path, Actions, total_episodes, hyperparameters):

    register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:RML_LetterEnv_5',
    max_episode_steps=200
    )
    env = environment(config_path)

    env.env.set_n(1)
    
    
    q_table = {}
    alpha, gamma, epsilon, eps_decay = hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    episode_rewards = []
    successes = []
    for episode in tqdm(range(total_episodes)):
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
            next_state, reward, done, _, __ = env.step(action)
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
            total_reward += reward

        # Decay epsilon
        epsilon *= eps_decay

        episode_rewards.append(total_reward)    # Only checking the last reward as it contains information on whether the episode is a success
        if reward >= 20:
            successes.append(episode)

    
    return episode_rewards, successes


def learning_episode_old(environment, config_path, Actions, total_episodes, hyperparameters):

    register(
    id='letter-env',
    entry_point='envs.letterenv_wrappers:Letter_RMLEnv_Old',
    max_episode_steps=200
    )
    env = environment(config_path)

    env.env.set_n(1)
    
    
    q_table = {}
    alpha, gamma, epsilon, eps_decay = hyperparameters[0], hyperparameters[1], hyperparameters[2], hyperparameters[3]
    actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]
    episode_rewards = []
    successes = []
    for episode in tqdm(range(total_episodes)):
        state , _ = env.reset()

        done = False
        total_reward = 0
        while not done:
            if state not in q_table:         # Add state to the q table
                q_table[state] = {action: 0 for action in actions}
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.choice(actions)
            else:
                max_value = max(q_table[state].values())             # Find all actions that have this maximum Q-value among valid actions
                best_actions = [a for a in actions if q_table[state][a] == max_value]
                action = random.choice(best_actions)                 # Randomly choose among the best actions

            # Take action
            next_state, reward, done, _, __ = env.step(action)

            if next_state not in q_table:    # Add next state to the q table
                q_table[next_state] = {action: 0 for action in actions}

            # Q-learning update
            old_value = q_table[state][action]
            next_max = max(q_table[next_state].values())
            q_table[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)

            state = next_state
            total_reward += reward

        # Decay epsilon
        epsilon *= eps_decay

        episode_rewards.append(total_reward)    # Only checking the last reward as it contains information on whether the episode is a success
        if reward >= 20:
            successes.append(episode)

    
    return episode_rewards, successes