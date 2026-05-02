def eval_agent(
    agent,
    env,
):
    agent.reset()
    obs, _ = env.reset()
    obs = env.observation(obs)

    for t in range(100):
        action = agent.get_greedy_action(obs)
        next_obs, reward, terminated, _, _ = env.step(action)
        next_obs = env.observation(next_obs)
        obs = next_obs

        agent.step(next_obs)

        if agent.terminated():
            break
    return reward
