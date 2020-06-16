"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import neurogym as ngym
import numpy as np
import warnings


class n_variable(ngym.TrialWrapper):
    metadata = {
        'description': 'Change number of active choices every ' +
        'block_dur trials. Always less or equal than original number.',
        'paper_link': None,
        'paper_name': None
    }

    def __init__(self, env, block_dur=10, prob_2=None):
        """
        block_dur: duration of each blok containing an specific number
        of active choices
        prob_2: probability of having only two active choices per block
        """
        super().__init__(env)

        assert isinstance(block_dur, int), 'block_dur must be integer'
        assert isinstance(self.task, ngym.TrialEnv), 'Task has to be TrialEnv'

        self.block_dur = block_dur
        self.max_n_ch = len(self.task.choices)  # Max number of choices

        # uniform distr. across choices unless prob(n_ch=2) (prob_2) is specified
        if prob_2 is not None:
            assert 0 <= prob_2 <= 1, 'Probability must be number within [0,1]'
            p = (1-prob_2)/(self.max_n_ch-2)
            self.prob = [p]*(self.max_n_ch-1)  # n_ch-1 possible values (2, 3,..)
            self.prob[0] = prob_2  # 2-choice blocks have prob_2 of occurrence
            self.prob /= np.sum(self.prob)  # We ensure total 1 sum.
        else:
            self.prob = [1/(self.max_n_ch-1)]*(self.max_n_ch-1)
        # Initialize with a random number of active choices (never 1)
        self.nch = self.rng.choice(self.task.choices[1:])

    def new_trial(self, **kwargs):

        if 'ground_truth' in kwargs.keys():
            warnings.warn('Warning: n_variable wrapper ' +
                          'will ignore passed ground truth')

        if self.task.num_tr % self.block_dur == 0:
            # We change number of active choices every 'block_dur'.
            self.nch = self.rng.choice(self.task.choices[1:], p=self.prob)

        kwargs.update({'n_ch': self.nch})
        self.env.new_trial(**kwargs)

    def step(self, action, new_tr_fn=None):
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        info['nch'] = self.nch
        return obs, reward, done, info
