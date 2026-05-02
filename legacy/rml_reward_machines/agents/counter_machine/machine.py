import numpy as np


class CounterMachine:
    def _init_transitions(self):
        curr_delta_u = self.delta_u.copy()
        curr_delta_c = self.delta_c.copy()
        curr_delta_r = self.delta_r.copy()
        change_made = True

        while change_made:
            change_made = False
            last_delta_u = curr_delta_u.copy()
            last_delta_c = curr_delta_c.copy()
            last_delta_r = curr_delta_r.copy()
            curr_delta_u = {}
            curr_delta_c = {}
            curr_delta_r = {}

            for k in last_delta_u:
                counter_states = k[2]

                try:
                    idx = counter_states.index("-")
                    counter_states_z = (
                        counter_states[:idx] + ("Z",) + counter_states[idx + 1 :]
                    )
                    counter_states_nz = (
                        counter_states[:idx] + ("NZ",) + counter_states[idx + 1 :]
                    )

                    key_z = (k[0], k[1], counter_states_z)
                    key_nz = (k[0], k[1], counter_states_nz)

                    if key_z not in last_delta_u:
                        curr_delta_u[key_z] = last_delta_u[k]
                        curr_delta_c[key_z] = last_delta_c[k]
                        curr_delta_r[key_z] = last_delta_r[k]
                    else:
                        curr_delta_u[key_z] = last_delta_u[key_z]
                        curr_delta_c[key_z] = last_delta_c[key_z]
                        curr_delta_r[key_z] = last_delta_r[key_z]

                    if key_nz not in last_delta_u:
                        curr_delta_u[key_nz] = last_delta_u[k]
                        curr_delta_c[key_nz] = last_delta_c[k]
                        curr_delta_r[key_nz] = last_delta_r[k]
                    else:
                        curr_delta_u[key_nz] = last_delta_u[key_nz]
                        curr_delta_c[key_nz] = last_delta_c[key_nz]
                        curr_delta_r[key_nz] = last_delta_r[key_nz]

                    change_made = True
                except ValueError:
                    try:
                        curr_delta_u[k] = last_delta_u[k]
                        curr_delta_c[k] = last_delta_c[k]
                        curr_delta_r[k] = last_delta_r[k]
                    except KeyError:
                        raise KeyError(
                            "Incorrect machine configuration. Check keys consistent"
                            "across delta_u, delta_c, and delta_r."
                        )

        self.delta_u = curr_delta_u
        self.delta_c = curr_delta_c
        self.delta_r = curr_delta_r
        if set(self.delta_u.keys()) != set(self.delta_c.keys()) or set(self.delta_u.keys()) != set(self.delta_r.keys()):
            raise KeyError("Mismatch in keys across delta_u, delta_c, and delta_r.")



    def transition(self, props, u, counters):
        counter_states = tuple("Z" if c == 0 else "NZ" for c in counters)
        key = (props, u, counter_states)

        next_u = self.delta_u[key]
        reward = self.delta_r[key]
        counter_delta = self.delta_c[key]

        u = next_u
        counters = tuple(np.add(counters, counter_delta))
        return u, counters, reward
