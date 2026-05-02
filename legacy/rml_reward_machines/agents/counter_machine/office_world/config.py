from agents.counter_machine.machine import CounterMachine

# The delta_u dictionary defines the transitions between states based on the current input, state, and counter condition
# an example -             (("C",), 2, ("Z",)): 2, (Z means zero for counter, NZ means non zero for counter and - is any value)
 # The delta_c dictionary defines how the counter is updated based on the current input, state, and counter condition
        
# The delta_r dictionary defines whether to reset the automaton based on the current input, state, and counter condition

class OfficeWorldCounterMachine(CounterMachine):
    def __init__(self):
        super().__init__()
        self.U = [0, 1, 2]  # States: 0 - Collecting e, 1 - At o, 2 - Collecting f, 3 - Visiting g
        self.F = [3]  # Final state, no further transitions
        self.u_0 = 0  # Initial state

        self.delta_u = {
            # State 0: Collect 'e' n times
            (("E",), 0, ("-","-")): 0,  # Stay in state 0 while collecting 'e'
            (("O",), 0, ("-","-")): 1,  # Transition to state 1 after collecting 'o'
            (("F",), 0, ("-","-")): 3,  # Terminate if 'f' is visited before 'o'
            (("G",), 0, ("-","-")): 3,  # Terminate if 'g' is visited before 'o'
            ((), 0, ("-","-")): 0,     
            (("A",), 0, ("-","-")): 0,      # Stay in state 0 otherwise
            (("B",), 0, ("-","-")): 0,      # Stay in state 0 otherwise
            (("C",), 0, ("-","-")): 0,      # Stay in state 0 otherwise
            (("D",), 0, ("-","-")): 0,      # Stay in state 0 otherwise
            (("N",), 0, ("-","-")): 3,      # Terminate at N

            # State 1: Collect 'f' n times
            (("E",), 1, ("-","-")): 3,  # Terminate if 'e' is revisited
            (("O",), 1, ("-","-")): 3,  # Terminate if 'o' is revisited
            (("F",), 1, ("Z","-")): 3,  # Can't visit F when the counter has been decremented to zero
            (("F",), 1, ("NZ","-")): 1, # Stay in state 1 while collecting 'f' with non-zero counter
            (("G",), 1, ("-","-")): 3,  # Transition to state 3 after finishing collection of 'f'
            ((), 1, ("Z","-")): 2,     # Stay in state 1 otherwise
            ((), 1, ("NZ","-")): 1,     # Stay in state 1 otherwise
            (("A",), 1, ("-","-")): 1,      # Stay in state 1 otherwise
            (("B",), 1, ("-","-")): 1,      # Stay in state 1 otherwise
            (("C",), 1, ("-","-")): 1,      # Stay in state 1 otherwise
            (("D",), 1, ("-","-")): 1,      # Stay in state 1 otherwise
            (("N",), 1, ("-","-")): 3,      # Terminate at N

            # State 2: Visit 'g' n times
            (("E",), 2, ("-","-")): 3,  # Terminate if 'e' is revisited
            (("O",), 2, ("-","-")): 3,  # Terminate if 'o' is revisited
            (("F",), 2, ("-","-")): 3,  # Terminate if 'f' is revisited
            (("G",), 2, ("-","Z")): 2,  # Stay in state 2 while collecting 'g' with zero counter
            (("G",), 2, ("-","NZ")): 2, # Stay in state 2 while collecting 'g' with non-zero counter
            ((), 2, ("Z","NZ")): 2,     # Stay in state 2 otherwise
            ((), 2, ("Z","Z")): 3,      # Terminate when 'g' is visited n times
            (("A",), 2, ("-","-")): 2,      # Stay in state 2 otherwise
            (("B",), 2, ("-","-")): 2,      # Stay in state 2 otherwise
            (("C",), 2, ("-","-")): 2,      # Stay in state 2 otherwise
            (("D",), 2, ("-","-")): 2,      # Stay in state 2 otherwise
            (("N",), 2, ("-","-")): 3,      # Terminate at N
        }

        self.delta_c = {
            # State 0: Increment the counter when visiting 'e'
            (("E",), 0, ("-","-")): (1,1),
            (("O",), 0, ("-","-")): (0,0),
            (("F",), 0, ("-","-")): (0,0),
            (("G",), 0, ("-","-")): (0,0),
            ((), 0, ("-","-")): (0,0),
            (("A",), 0, ("-","-")): (0,0),      
            (("B",), 0, ("-","-")): (0,0),      
            (("C",), 0, ("-","-")): (0,0),      
            (("D",), 0, ("-","-")): (0,0),      
            (("N",), 0, ("-","-")): (0,0),      

            # State 1: Decrement the counter when visiting 'f'
            (("E",), 1, ("-","-")): (0,0),
            (("O",), 1, ("-","-")): (0,0),
            (("F",), 1, ("Z","-")): (0,0),
            (("F",), 1, ("NZ","-")): (-1,0),
            (("G",), 1, ("-","-")): (0,0),
            ((), 1, ("Z","-")): (0,0),
            ((), 1, ("NZ","-")): (0,0),
            (("A",), 1, ("-","-")): (0,0),      
            (("B",), 1, ("-","-")): (0,0),      
            (("C",), 1, ("-","-")): (0,0),      
            (("D",), 1, ("-","-")): (0,0),      
            (("N",), 1, ("-","-")): (0,0),      

            # State 2: No counter changes at 'g'
            (("E",), 2, ("-","-")): (0,0),
            (("O",), 2, ("-","-")): (0,0),
            (("F",), 2, ("-","-")): (0,0),
            (("G",), 2, ("-","Z")): (0,0),
            (("G",), 2, ("-","NZ")): (0,-1),
            ((), 2, ("Z","NZ")): (0,0),
            ((), 2, ("Z","Z")): (0,0),
            (("A",), 2, ("-","-")): (0,0),      
            (("B",), 2, ("-","-")): (0,0),      
            (("C",), 2, ("-","-")): (0,0),      
            (("D",), 2, ("-","-")): (0,0),      
            (("N",), 2, ("-","-")): (0,0),      
        }

        self.delta_r = {
            # State 0: No reset conditions
            (("E",), 0, ("-","-")): 1,
            (("O",), 0, ("-","-")): 1,
            (("F",), 0, ("-","-")): -1,
            (("G",), 0, ("-","-")): -1,
            ((), 0, ("-","-")): 0,
            (("A",), 0, ("-","-")): 0,      
            (("B",), 0, ("-","-")): 0,      
            (("C",), 0, ("-","-")): 0,      
            (("D",), 0, ("-","-")): 0,      
            (("N",), 0, ("-","-")): -1,      

            # State 1: No reset conditions
            (("E",), 1, ("-","-")): -1,
            (("O",), 1, ("-","-")): -1,
            (("F",), 1, ("Z","-")): -1,
            (("F",), 1, ("NZ","-")): 1,
            (("G",), 1, ("-","-")): -1,
            ((), 1, ("Z","-")): 0,
            ((), 1, ("NZ","-")): 0,
            (("A",), 1, ("-","-")): 0,      
            (("B",), 1, ("-","-")): 0,      
            (("C",), 1, ("-","-")): 0,      
            (("D",), 1, ("-","-")): 0,      
            (("N",), 1, ("-","-")): -1,  

            # State 2: No reset conditions
            (("E",), 2, ("-","-")): -1,
            (("O",), 2, ("-","-")): -1,
            (("F",), 2, ("-","-")): -1,
            (("G",), 2, ("-","NZ")): 1,
            (("G",), 2, ("-","Z")): 1,
            ((), 2, ("Z","NZ")): 0,
            ((), 2, ("Z","Z")): 1,  # Reset if counter reaches zero
            (("A",), 2, ("-","-")): 0,      
            (("B",), 2, ("-","-")): 0,      
            (("C",), 2, ("-","-")): 0,      
            (("D",), 2, ("-","-")): 0,      
            (("N",), 2, ("-","-")): -1,  
        }

        self._init_transitions()





        