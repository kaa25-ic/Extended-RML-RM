import random, math, os
import numpy as np
from enum import Enum
import random
from gymnasium import spaces

"""
Enum with the actions that the agent can execute
"""
class Actions(Enum):
    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left
    none  = 4 # none or pick
    drop  = 5


class OfficeWorld:

    up    = 0 # move up
    right = 1 # move right
    down  = 2 # move down
    left  = 3 # move left

    def __init__(self,
                 propositions=["A","B","C","D","E","F","G","N","O"],
                 task_string = "EEEEEOFFFFFGGGGG",
                 max_observation_counts: dict[str, int | None] = {
            "E": 5,
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "F": None,
            "G": None,
            "N": None,
            "O": None
        },
        replacement_mapping: dict[str, str] = {"E": "O"},
        max_episode_steps= 500
        ):
        self._load_map()
        self.n_rows, self.n_cols = 9, 12
        self.additional_states()
        self.propositions = propositions
        self.task_string = task_string
        self.max_observation_counts = max_observation_counts
        self.replacement_mapping = replacement_mapping
        self.prop_idx = {p: i for i, p in enumerate(self.propositions)}
        self.max_episode_steps = max_episode_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple(
            spaces=(
                spaces.Discrete(self.n_rows),
                spaces.Discrete(self.n_cols),
                spaces.Text(
                    min_length=0,
                    max_length=1,
                    charset="".join(self.propositions + ["_"]),
                ),
            ),
        )
        self.reward_range = (0, 1)




    def additional_states(self):
        self.state_objects = {}
        for key, value in self.objects.items():
            if value != 'N':
                self.state_objects[key] = value
        self.object_list = self.remove_duplicate_f_from_list(list(self.state_objects.values()))
    
    def remove_duplicate_f_from_list(self, original_list):
        has_f = False
        result_list = []
        for item in original_list:
            if item == 'F':
                if not has_f:  # If 'f' hasn't been added yet
                    result_list.append(item)
                    has_f = True  # Set the flag to True after adding 'f' once
            else:
                result_list.append(item)  # Add all non-'f' items normally
        return result_list
    
    def get_additional_state_vector(self):
        self.true_props = self.get_true_propositions()
        new_objects = [0]*len(self.object_list)    # The new list for the one hot encoding

        if 'F' in self.object_list:  # remembering f as it remains after leaving the f position
            f_index = self.object_list.index('F')

        if len(self.true_props) > 0:   # Adding 1s to where currently possesed objects are
            if self.true_props in self.object_list:
                encoding_index = self.object_list.index(self.true_props)
                if self.one_hot_objects[f_index] == 1:
                    new_objects[encoding_index] = 1
                    new_objects[f_index] = 1      
                else:
                    new_objects[encoding_index] = 1
        return new_objects
     

    def reset(self, seed=None, options=None):
        self.n_steps = 0
        self.task_string_idx = 0
        self.task_failed = False
        self.prop_obs_counts = np.zeros((len(self.propositions),))
        self.agent = (1,0)
        self.objects[(7,4)] = "E"  # MAIL


        self.true_props = self.get_true_propositions()
        self.one_hot_objects = [0]*len(self.object_list)
        self.one_hot_objects = self.get_additional_state_vector()
    
        return (self.agent[0],self.agent[1],self.true_props), {}

    def step(self, a):
        """
        We execute 'action' in the game
        """
        self.n_steps += 1
        x,y = self.agent
        self.agent = self._get_new_position(x,y,a)
        obs = self.get_features()
        obs_prop = obs[2]
        # Calculate which propositions are true in the environment
        if obs_prop != "_":
            # Update number of times proposition has been observed
            prop_idx = self.prop_idx[obs_prop]
            self.prop_obs_counts[prop_idx] += 1

            if self.prop_obs_counts[prop_idx] == self.max_observation_counts[obs_prop]:
                # Delete proposition
                self.objects[self.agent] = self.replacement_mapping[obs_prop]

            try:
                if obs_prop == self.task_string[self.task_string_idx]:
                    self.task_string_idx += 1
                else:
                    if obs_prop not in ['A','B','C','D']:
                        self.task_failed = True


            except IndexError:
                pass
        # Determine if episode is terminated due to max number of steps
        if self.max_episode_steps == self.n_steps:
            terminated = True
            # Episode ended based on condition outside MDP (interface for more details)
            truncated = True
            reward = 0
        else:
            if self.task_failed:
                terminated = True
                truncated = False
                reward = 0
            elif self.task_string_idx == len(self.task_string):
                terminated = True
                truncated = False
                reward = 1
            else:
                terminated = False
                truncated = False
                reward = 0
        return (
            obs,
            reward,
            terminated,
            truncated,
            {},
        )

    def _get_new_position(self, x, y, a):
        action = Actions(a)
        # executing action
        if (x,y,action) not in self.forbidden_transitions:
            if action == Actions.up   : y+=1
            if action == Actions.down : y-=1
            if action == Actions.left : x-=1
            if action == Actions.right: x+=1
        return x,y


    def get_true_propositions(self):
        """
        Returns the string with the propositions that are True in this state
        """
        ret = ""
        if self.agent in self.objects:
            ret += self.objects[self.agent]
        else:
            ret += "_"
        return ret

    def get_features(self):
        """
        Returns the features of the current state (i.e., the location of the agent)
        """
        x,y = self.agent
        self.one_hot_objects = self.get_additional_state_vector()
        self.true_props = self.get_true_propositions()
        return (x,y,self.true_props)

    def show(self):
        for y in range(8,-1,-1):
            if y % 3 == 2:
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.up) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                
            for x in range(12):
                if (x,y,Actions.left) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 0:
                    print(" ",end="")
                if (x,y) == self.agent:
                    print("A",end="")
                elif (x,y) in self.objects:
                    print(self.objects[(x,y)],end="")
                else:
                    print(" ",end="")
                if (x,y,Actions.right) in self.forbidden_transitions:
                    print("|",end="")
                elif x % 3 == 2:
                    print(" ",end="")
            print()      
            if y % 3 == 0:      
                for x in range(12):
                    if x % 3 == 0:
                        print("_",end="")
                        if 0 < x < 11:
                            print("_",end="")
                    if (x,y,Actions.down) in self.forbidden_transitions:
                        print("_",end="")
                    else:
                        print(" ",end="")
                print()                

    def get_model(self):
        """
        This method returns a model of the environment. 
        We use the model to compute optimal policies using value iteration.
        The optimal policies are used to set the average reward per of each task to 1.
        """
        S = [(x,y) for x in range(12) for y in range(9)] # States
        A = self.actions.copy() # Actions
        L = self.objects.copy() # Labeling function
        T = {}                  # Transitions (s,a) -> s' (they are deterministic)
        for s in S:
            x,y = s
            for a in A:
                T[(s,a)] = self._get_new_position(x,y,a)
        return S,A,L,T # SALT xD

    def _load_map(self):
        # Creating the map
        self.objects = {}
        self.objects[(1,1)] = "A"
        self.objects[(1,7)] = "B"
        self.objects[(10,7)] = "C"
        self.objects[(10,1)] = "D"
        self.objects[(7,4)] = "E"  # MAIL
        self.objects[(8,2)] = "F"  # COFFEE
        self.objects[(3,6)] = "F"  # COFFEE
        self.objects[(4,4)] = "G"  # OFFICE
        self.objects[(4,1)] = "N"  # PLANT
        self.objects[(7,1)] = "N"  # PLANT
        self.objects[(4,7)] = "N"  # PLANT
        self.objects[(7,7)] = "N"  # PLANT
        self.objects[(1,4)] = "N"  # PLANT
        self.objects[(10,4)] = "N" # PLANT
        # Adding walls
        self.forbidden_transitions = set()
        # general grid
        for x in range(12):
            for y in [0,3,6]:
                self.forbidden_transitions.add((x,y,Actions.down)) 
                self.forbidden_transitions.add((x,y+2,Actions.up))
        for y in range(9):
            for x in [0,3,6,9]:
                self.forbidden_transitions.add((x,y,Actions.left))
                self.forbidden_transitions.add((x+2,y,Actions.right))
        # adding 'doors'
        for y in [1,7]:
            for x in [2,5,8]:
                self.forbidden_transitions.remove((x,y,Actions.right))
                self.forbidden_transitions.remove((x+1,y,Actions.left))
        for x in [1,4,7,10]:
            self.forbidden_transitions.remove((x,5,Actions.up))
            self.forbidden_transitions.remove((x,6,Actions.down))
        for x in [1,10]:
            self.forbidden_transitions.remove((x,2,Actions.up))
            self.forbidden_transitions.remove((x,3,Actions.down))
        # Adding the agent
        self.actions = [Actions.up.value,Actions.right.value,Actions.down.value,Actions.left.value]

    @property
    def unwrapped(self):
        """Return the 'raw' environment with no wrappers."""
        return self
    


class OfficeWorld_Delivery(OfficeWorld):

    def __init__(self,
                 propositions=["A","B","C","D","E","F","G","N"],
                 task_string = "EFG",
                 max_observation_counts = {
            "E": 1,
            "A": None,
            "B": None,
            "C": None,
            "D": None,
            "F": None,
            "G": None,
            "N": None
        },
        replacement_mapping = {"E": "O"},
        max_episode_steps = 200
        ):
        super().__init__(
                 propositions= propositions,
                 task_string = task_string,
                 max_observation_counts = max_observation_counts,
        replacement_mapping = replacement_mapping,
        max_episode_steps= max_episode_steps
        )
    
    def get_additional_state_vector(self):
            self.true_props = self.get_true_propositions()
            new_objects = [0]*len(self.object_list)    # The new list for the one hot encoding

            e_index = self.object_list.index('E')

            if 'F' in self.object_list:  # remembering f as it remains after leaving the f position
                f_index = self.object_list.index('F')

            if len(self.true_props) > 0:   # Adding 1s to where currently possesed objects are
                if self.true_props in self.object_list:
                    encoding_index = self.object_list.index(self.true_props)
                    if self.one_hot_objects[f_index] == 1:
                        new_objects[encoding_index] = 1
                        new_objects[f_index] = 1     # Making a random number of coffees collected between a and b
                    elif 'E' in self.true_props:
                        new_objects[encoding_index] = 1
                        new_objects[e_index] = random.randint(1,3)
                    else:
                        new_objects[encoding_index] = 1

            return new_objects
    
    
