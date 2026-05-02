import os
import pickle
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
from envs.letterenv import Actions
import random
from tqdm import tqdm

import numpy as np
from agents.counter_machine.agent import (
    CounterMachineCRMAgent_NewAction_ExtraReward
)

from agents.counter_machine.context_free.config_conditional import (
    ContextFreeCounterMachine_Conditional
)

from environments.context_free_conditional import (
    create_context_free_env_conditional_labelled,
    labelled_action_space,
    mdp_action_space,
)
from utils.train import train_letter_conditional


agent_kwargs = {
    "initial_epsilon": 0.35,
    "final_epsilon": 0.01,
    "epsilon_decay": 0.995,
    "learning_rate": 0.5,
    "discount_factor": 0.99,
}

def create_counter_crm_agent():
    return CounterMachineCRMAgent_NewAction_ExtraReward(
        machine=ContextFreeCounterMachine_Conditional(),
        action_space=labelled_action_space,
        **agent_kwargs,
    )


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")

    n = 4

    print(f"Training started")
    agent = create_counter_crm_agent()
    agent.epsilon_decay = agent_kwargs["epsilon_decay"]

    convergence_results = []
    for i in tqdm(range(1, 21)):
        conv_results = train_letter_conditional(Actions, agent, n_episodes=100000,N=n)
        print(conv_results)
        convergence_results.append(conv_results)

    print(convergence_results)
    
    with open(f"results/convergence_results_CRA_letterenv_conditional.pkl", "wb") as f:
        pickle.dump(convergence_results, f)



