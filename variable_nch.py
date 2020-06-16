"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import neurogym as ngym
import numpy as np
import warnings


class Variable_nch(ngym.TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_nch trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, block_nch=100, prob_2=None):
        """
        block_nch: duration of each blok containing an specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)

        assert isinstance(block_nch, int), 'block_nch must be integer'
        assert isinstance(self.task, ngym.TrialEnv), 'Task has to be TrialEnv'

        self.block_nch = block_nch
        self.max_nch = len(self.task.choices)  # Max number of choices

        # uniform distr. across choices unless prob(n_ch=2) (prob_2) is specified
        if prob_2 is not None:
            assert 0 <= prob_2 <= 1, 'Probability must be number within [0,1]'
            p = (1-prob_2)/(self.max_nch-2)
            self.prob = np.append(prob_2, [p]*(self.max_nch-2))
            self.prob /= np.sum(self.prob)  # We ensure total 1 sum.
        else:
            self.prob = [1/(self.max_nch-1)]*(self.max_nch-1)
        # Initialize with a random number of active choices (never 1)
        self.nch = self.rng.choice(range(2, self.max_nch + 1))

    def new_trial(self, **kwargs):

        if 'ground_truth' in kwargs.keys():
            warnings.warn('Variable_nch wrapper ' +
                          'will ignore passed ground truth')

        if self.task.num_tr % self.block_nch == 0:
            # We change number of active choices every 'block_nch'.
            self.nch = self.rng.choice(range(2, self.max_nch + 1),
                                        p=self.prob)

        kwargs.update({'n_ch': self.nch})
        self.env.new_trial(**kwargs)

    def step(self, action, new_tr_fn=None):
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        info['nch'] = self.nch
        return obs, reward, done, info
