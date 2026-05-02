class RewardMachine:
    def transition(self, props, u):
        u_prime = self.delta_u[(props, u)]
        reward = self.delta_r[(props, u)]
        return u_prime, reward
