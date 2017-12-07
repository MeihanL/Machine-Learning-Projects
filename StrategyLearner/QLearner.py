"""Author: Liu Meihan"""

import numpy as np
import random as rand

class QLearner(object):
    """
        Q learning with Dyna-Q solutions

        num_states integer, the number of states to consider
        num_actions integer, the number of actions available.
        alpha float, the learning rate used in the update rule. Should range between 0.0 and 1.0
        gamma float, the discount rate used in the update rule. Should range between 0.0 and 1.0
        rar float, random action rate: the probability of selecting a random action at each step.
                   Should range between 0.0 (no random actions) to 1.0 (always random action)
        radr float, random action decay rate, after each update, rar = rar * radr
                    Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay)
        dyna integer, conduct this number of dyna updates for each regular update.
        verbose boolean, if True, your class is allowed to print debugging statements,
                         if False, all printing is prohibited.

    """
    def __init__(self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False):
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Qtable = np.zeros((num_states, num_actions))
        self.counter = 3000
        if self.dyna > 0:
            self.Tc = np.ndarray((num_states, num_actions, num_states))
            self.Tc.fill(0.00000001)
            self.T = self.Tc/self.Tc.sum(axis = 2, keepdims=True)
            self.R = np.zeros((num_states, num_actions))
            # self.R.fill(-1.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        random_num = rand.random()
        if random_num <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            best_q_value = max(self.Qtable[s])
            indices = [index for index , q in enumerate(self.Qtable[s]) if q == best_q_value]
            action = rand.choice(indices)
        self.a = action
        if self.verbose: print "s =", s,"a =", action
        return action

    def query(self, s_prime, r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        best_q_prime = max(self.Qtable[s_prime,:])
        indices = [index for index, q in enumerate(self.Qtable[s_prime]) if q == best_q_prime]
        action_prime = rand.choice(indices)
        self.Qtable[self.s][self.a] = (1 - self.alpha) * self.Qtable[self.s][self.a]\
                                      + self.alpha * (r + self.gamma * self.Qtable[s_prime][action_prime])
        if self.dyna > 0:
            self.Tc[self.s, self.a, s_prime] += 1
            self.T[self.s, self.a,] = self.Tc[self.s, self.a,]/self.Tc[self.s, self.a,].sum()
            self.R[self.s, self.a] = (1- self.alpha)*self.R[self.s,self.a]+self.alpha*r
            if self.counter > 0:
                self.counter -= 1
            else:
                rand_s = np.random.randint(0, self.num_states, size=self.dyna)
                rand_a = np.random.randint(0, self.num_actions, size=self.dyna)
                # sample s_prime based on probability
                s_dyna = [np.random.multinomial(1, self.T[rand_s[i], rand_a[i],]) for i in range(self.dyna)]
                s_dyna = np.array(s_dyna)
                s_prime_dyna = np.where(s_dyna == 1)[1]
                r = self.R[rand_s, rand_a]
                for i in range(self.dyna):
                    self.Qtable[rand_s[i], rand_a[i]] = (1 - self.alpha) * self.Qtable[rand_s[i], rand_a[i]]\
                                    + self.alpha * (r[i] + self.gamma * np.max(self.Qtable[s_prime_dyna[i],]))
        random_num = rand.random()
        if random_num <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
        else:
            best_q_value = max(self.Qtable[s_prime])
            indices = [index for index, q in enumerate(self.Qtable[s_prime]) if q == best_q_value]
            action = rand.choice(indices)
        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Q Learner"