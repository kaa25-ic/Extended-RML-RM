from agents.counter_machine.machine import CounterMachine

class RandomObjectsEnvCounterMachine(CounterMachine):
    def __init__(self):
        super().__init__()
        self.U = [0]  # States: 0 - Collecting objects
        self.F = [1]  # Final state, all objects collected, no further transitions
        self.u_0 = 0  # Initial state

        self.delta_u = {
            ((0,), 0, ("-","-","-")): 0, 
            ((1,), 0, ("-","-","-")): 0,
            ((2,), 0, ("-","-","-")): 0,  
            ((3,), 0, ("-","-","-")): 0,   
        }

        self.delta_c = {
            ((0,), 0, ("-","-","-")): (0,0,0), 
            ((1,), 0, ("-","-","-")): (1,0,0), 
            ((2,), 0, ("-","-","-")): (0,1,0),   
            ((3,), 0, ("-","-","-")): (0,0,1),     
        }

        self.delta_r = {     
            ((0,), 0, ("-","-","-")): -0.1, 
            ((1,), 0, ("-","-","-")): 1,
            ((2,), 0, ("-","-","-")): 1,  
            ((3,), 0, ("-","-","-")): 1, 
        }

        self._init_transitions()





        