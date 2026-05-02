# Introduction

This repository contains the code for the experiments related to **RML Reward Machines** [1].

# Setup

To run RML Reward Machines, you must have **RML** installed and running with a WebSocket connection.  
RML can be found at the official website, which includes downloads and introductory guides:  
👉 [https://rmlatdibris.github.io/](https://rmlatdibris.github.io/).

RML Reward Machines use custom monitors, located under `rml/rml_monitors`. To run the experiments, these monitors need to be copied into your local RML directory. 

If you are using a Windows system, note that RML requires a Linux-style environment to run. **Cygwin64** can be used for this purpose.

The experiments are implemented in **Python**, which must also be installed on your machine.

Once the prerequisites are in place, clone this repository and install the dependencies:

```bash
git clone https://github.com/danieldonnelly7/rml_reward_machines.git
cd rml_reward_machines
pip install -r requirements.txt
```

# Repository Structure
```
rml_reward_machines/
│
├── agents/                   # Contains a variety of reward machine based agents
│   ├── counter_machine       # Agents based on counting reward automata
│   ...                       # Other reward machine based agents
│
├── environments/             # Wrappers for standard reward machine environments
│
├── envs/                     # Custom environments and wrappers for RML Reward Machines
│
├── examples/                 # Example configuration files for RML Reward Machines
│
├── results/                  # Outputs of experiments
├── rml/                      # Core RML components
│   ├── rml_monitors/         # RML monitor definitions
│   ├── rml_specifications/   # RML specification files
│   ├── rmlgym.py             # RML Reward Machines
│   └── rmlgym_original.py    # Original RMLGym  
│
├── utils/                    # Helper utilities and shared functions
│
├── requirements.txt          # Python dependencies
├── LICENSE                   # Project license
├── README.md                 # Documentation
└── .gitignore                # Ignored files and folders
```

# Instructions

### Paper Experiments

This section contains instructions to run the experiments used in the paper [1]. The monitors required to perform the tasks are located in `rml/rml_monitors`,  and the corresponding specifications can be found in `rml/rml_specifications`.

#### Numerical Experiment — Comparison with Counting Reward Automata (CRA)

This experiment is implemented in `experiments/letterenv_numerical_cra_comparison.py` and uses the RML specification  `rml/rml_specifications/letter_env_spec_numerical.py`. 

To run this experiment:

1. **Copy the RML specification**  
   Copy `rml/rml_specifications/letter_env_spec_numerical.py` into your local RML directory.

2. **Start the RML monitor**  
   Open a Linux terminal, navigate to your local RML directory, and run:  
    ```bash
    sh ./online_monitor_edit.sh ./letter_env_spec_numerical.pl 8081
    ```

3. **(Optional) Save experiment results**  
     Uncomment the bottom lines of the experiment file experiments/letterenv_numerical_cra_comparison.py, and specify output filenames as appropriate.

4. **Run the experiment**  
    From the rml_reward_machines project directory, run:
    ```bash
    python -m experiments.letterenv_numerical_cra_comparison
    ```

5. **Generate Plot**  
    To recreate the graph, run the plotting script running plot_numerical_experiment_final.py under the results folder. 

#### Numerical Experiment — Comparison with RMLGym

This experiment is implemented across two files:

1. `experiments/letterenv_numerical_rmlrm_reward_per_episode.py` — runs the **RML Reward Machine** component  
2. `experiments/letterenv_numerical_rmlgym_test_run.py` — runs the **RMLGym** component  

Both use the RML specification `rml/rml_specifications/letter_env_spec_numerical.py`.

The components of the experiment can be run in either order. To run this experiment:

1. **Copy the RML specification**  
   Copy `rml/rml_specifications/letter_env_spec_numerical.py` into your local RML directory.

2. **(Optional) Save experiment results**  
   Uncomment the bottom lines of both experiment files and specify output filenames as appropriate.

3. **Run RML Reward Machines Component**  
   **Start Monitor**  
   Open a Linux terminal, navigate to your local RML directory, and run:  
   ```bash
   sh ./online_monitor_edit.sh ./letter_env_spec_numerical.pl 8081
   ```
   **Run experiment**  
   From the `rml_reward_machines` project directory, run:
   ```bash
   python -m experiments.letterenv_numerical_rmlrm_reward_per_episode
   ```

4. **Run RMLGym component**  
   
   **Start Monitor**  
    Open a Linux terminal, navigate to your local RML directory, and run:  
    ```bash
    sh ./online_monitor_original_terminate.sh ./letter_env_spec_numerical.pl 8081
    ```

   **Run experiment**   
    From the `rml_reward_machines` project directory, run:
    ```bash
    python -m experiments.letterenv_numerical_rmlgym_test_run
    ```

5. **Generate Plot**  
    To recreate the graph, run the plotting script running plot_rml_successes_comparison.py under the results folder. 


# References

[1] Donnelly, D., Ferrando, A., & Belardinelli, F. (2025). Expressive Reward Synthesis with the Runtime Monitoring Language. arXiv:2510.16185. https://arxiv.org/abs/2510.16185