import re
from collections import defaultdict
import numpy as np


def replace_numerical_parts(event):
    # Replace numbers and expressions like [3], [1+1], [1.0-1], [5.0], [1.0,2.0], etc. with the placeholder [{num}]
    # Handles comma-separated values as well as individual expressions
    return re.sub(r'\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*?)\]', lambda m: '[' + ','.join('{num}' for _ in m.group(1).split(',')) + ']', event)


# Function to extract numerical values from a part
def extract_numerical_value(part):
    # Match numerical expressions, including comma-separated values
    matches = re.findall(r'\[(\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*(?:,\d+(\.\d+)?(?:\+\d+(\.\d+)?|\-\d+(\.\d+)?)*?)*)\]', part)
    values = []
    for match in matches:
        # Split the matched expression by commas
        expressions = match[0].split(',')
        for expression in expressions:
            try:
                # Evaluate each expression
                value = eval(expression)
                values.append(value)
            except Exception as e:
                # Handle potential errors in evaluation
                print(f"Error evaluating expression '{expression}': {e}")
    if values:
        values = [0.01 if value == 0 else value for value in values]   # Replacing instances of 0 with 1 to differentiate from the not relevant case
    else:
        values = None
    return values


def event_string_replace(state_str):
    """
    Removes redundant characters from the state string and splits into a list of events in sequence.
    """
    state_str = state_str.replace('@', '')  # Remove '@' sign
    if state_str.startswith('(eps'):   # A lot of sub states start with eps, which doesn't have semantic value so removing it
        state_str = state_str[len('(eps*'):]
    parts = state_str.split('*')   # Splitting on * term as this separates different event types
    return parts

# Function to extract event types separated by '*'
def extract_events(state_str):
    parts = event_string_replace(state_str)
    events = [replace_numerical_parts(part.strip()) for part in parts]    # getting a list of events, with numerical parts turned to {num} to make sure they are encoded on the same index for different numerical values
    return events

def generate_events_and_index(states):
# Extract all unique events
    unique_events = set()
    for state in states.values():
        events = extract_events(state)
        unique_events.update(events) 
    unique_events = list(unique_events)
    i = 0
    event_index = {}

    for event in unique_events:
        event_index[event] = i
        if event.count('{num}') > 1:
            for j in range(1,event.count('{num}')):
                i += 1
                event_index[event + '£ADDITIONAL£'*j] = i    # For each additional {num} term adding an additional term to the string end. Number of these additional terms can be used to determine the appropriate {num} term for index
        i += 1
    return unique_events, event_index
#event_index = {event: idx for idx, event in enumerate(unique_events)} # Old logic just in case I want to keep it

def encode_state_in_vector(vector, part, values, event_index):
    """
    Function generates an encoding for a state (called part) in the output vector
    """
    if part in event_index:          # Inputting value at correct index
        if values is None:
            vector[event_index[part]] = 1
        else:       # Logic that adds in the value of each numerical value at the appropriate index
            add_elements = 0
            for value in values:
                vector[event_index[part + '£ADDITIONAL£'*add_elements]] = value
                add_elements += 1
    else:
        print('UNKNOWN State')  
    return vector

# Function to create one-hot encoding vectors considering sequences
def create_encoding(state_str, event_index):
    """
    Function that creates the encoding vector for a given state string.
    """

    parts = event_string_replace(state_str)
    part = parts[0]
    values = extract_numerical_value(part)     # Getting the numerical value to use for the index. f it's 0 using a small numerical value to distinguish from the case where the state is inactive

    vector = np.zeros(len(event_index))
    part = replace_numerical_parts(part)  # Replacing numerical part so it matches the relevant sub state
    vector = encode_state_in_vector(vector, part, values, event_index)

    if 'star' in part and 1 < len(parts):    # Handling cases where star is the next state
        next_part = parts[1]                
        
        next_values = extract_numerical_value(next_part)     # Using the same logic as above, extracting the value, replacing numerical part etc

        next_part = replace_numerical_parts(next_part)
        vector = encode_state_in_vector(vector, next_part, next_values, event_index)
    
    return vector


def generate_events_and_index_one_hot(the_states):
    # Extract all unique events
    unique_events = set()
    for state in the_states.values():
        events = extract_events(state)
        unique_events.update(events) 
    unique_events = list(unique_events)
    i = 0
    event_index = {}

    for event in unique_events:
        event_index[event] = i
        i += 1
    return unique_events, event_index

def encode_state_in_vector_one_hot(vector, part, event_index):
    """
    Function generates an encoding for a state (called part) in the output vector.

    This is the non numerical version (i.e. everything encoded as a 1)
    """
    if part in event_index:          # Inputting value at correct index
        vector[event_index[part]] = 1
    else:
        print('UNKNOWN State')  
    return vector

def create_encoding_one_hot(state_str, event_index):
    """
    Function that creates the encoding vector for a given state string.
    """

    parts = event_string_replace(state_str)
    part = parts[0]

    vector = np.zeros(len(event_index))
    part = replace_numerical_parts(part)  # Replacing numerical part so it matches the relevant sub state
    vector = encode_state_in_vector_one_hot(vector, part, event_index)

    if 'star' in part and 1 < len(parts):    # Handling cases where star is the next state
        next_part = parts[1]                
        
        next_part = replace_numerical_parts(next_part)
        vector = encode_state_in_vector_one_hot(vector, next_part, event_index)
    
    return vector

def create_encoding_RNN(state_str, event_index):
    """
    Function that creates the encoding vector for a given state string.
    """

    parts = event_string_replace(state_str) 
    vector = []
    for part in parts:
        new_vector = create_encoding(part,event_index)
        vector.append(new_vector)
    vector = np.array(vector)
    return vector