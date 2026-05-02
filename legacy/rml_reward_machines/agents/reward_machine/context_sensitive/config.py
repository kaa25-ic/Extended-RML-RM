from agents.reward_machine.machine import RewardMachine


class ContextSensitiveRewardMachine(RewardMachine):
    def __init__(self, N: int = 1):
        self.n_states = 3 * N + 1

        self.U = list(range(self.n_states))
        self.F = [self.n_states]
        self.u_0 = 0

        e_state_idx = N

        self.delta_u = {
            (("A",), e_state_idx): self.n_states,
            (("B",), e_state_idx): self.n_states,
            (("C",), e_state_idx): self.n_states,
            (("E",), e_state_idx): e_state_idx + 1,
            ((), e_state_idx): e_state_idx,
        }
        self.delta_r = {
            (("A",), e_state_idx): 0,
            (("B",), e_state_idx): 0,
            (("C",), e_state_idx): 0,
            (("E",), e_state_idx): 0,
            ((), e_state_idx): 0,
        }

        for n in range(N):
            a_state_idx = n
            b_state_idx = n + N + 1
            c_state_idx = b_state_idx + N

            delta_u_a = {
                (("A",), a_state_idx): a_state_idx + 1,
                (("B",), a_state_idx): self.n_states,
                (("C",), a_state_idx): self.n_states,
                (("E",), a_state_idx): self.n_states,
                ((), a_state_idx): a_state_idx,
            }
            delta_u_b = {
                (("A",), b_state_idx): self.n_states,
                (("B",), b_state_idx): b_state_idx + 1,
                (("C",), b_state_idx): self.n_states,
                (("E",), b_state_idx): self.n_states,
                ((), b_state_idx): b_state_idx,
            }
            delta_u_c = {
                (("A",), c_state_idx): self.n_states,
                (("B",), c_state_idx): self.n_states,
                (("C",), c_state_idx): c_state_idx + 1,
                (("E",), c_state_idx): self.n_states,
                ((), c_state_idx): c_state_idx,
            }
            delta_r_a = {
                (("A",), a_state_idx): 0,
                (("B",), a_state_idx): 0,
                (("C",), a_state_idx): 0,
                (("E",), a_state_idx): 0,
                ((), a_state_idx): 0,
            }
            delta_r_b = {
                (("A",), b_state_idx): 0,
                (("B",), b_state_idx): 0,
                (("C",), b_state_idx): 0,
                (("E",), b_state_idx): 0,
                ((), b_state_idx): 0,
            }
            delta_r_c = {
                (("A",), c_state_idx): 0,
                (("B",), c_state_idx): 0,
                (("C",), c_state_idx): 0,
                (("E",), c_state_idx): 0,
                ((), c_state_idx): 0,
            }

            self.delta_u |= delta_u_a
            self.delta_u |= delta_u_c
            self.delta_u |= delta_u_b
            self.delta_r |= delta_r_a
            self.delta_r |= delta_r_b
            self.delta_r |= delta_r_c

        self.delta_r[(("C",), self.n_states - 1)] = 1
