from envs.letterenv import Actions
from gymnasium.envs.registration import register
from rml.rmlgym_original import RMLGym
from utils.learning_functions import learning_episode_letter_original
from utils.train_rml import rml_training_original_hyper
import pickle


config_path = './examples/letter_env.yaml'

register(
    id='letter-env',
    entry_point='envs.property_envs.letterenv_numerical_wrappers:RML_LetterEnv_numerical_Original',
    max_episode_steps=200
)

env = RMLGym(config_path)

actions = [Actions.RIGHT.value, Actions.LEFT.value, Actions.UP.value, Actions.DOWN.value]

training_class = rml_training_original_hyper(learning_episode_letter_original, RMLGym, actions, config_path, n=1)

results = training_class.get_test_statistics()

top5 = results.sort_values(by="total successes", ascending=False).head(5)
print("Top 5 hyperparameter combos:")
print(top5)

with open('results/results_rml_original_hyperparam_test.pkl', 'wb') as f:
    pickle.dump(results, f)

# best result
# epsilon = 0.75, alpha = 0.01, gamma = 0.9, decay = 0.999, total successes = 9 (second best was 7)
# Note: best ones had similar parameters (high epsilon, low alpha, 0.999 decay)