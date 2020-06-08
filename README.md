# FinalDegreeProject
Wrap-up of my work developed during my Final Degree Project on Computational Neuroscience. 

```
from gym import spaces
import neurogym as ngym

class TrialWrapper(gym.Wrapper):
    """Base class for wrapping TrialEnv"""

    def __init__(self, env):
        super().__init__(env)
        self.env = env
        self.task = self.unwrapped

    def new_trial(self, **kwargs):
        self.env.new_trial()

    def reset(self, new_tr_fn=None, step_fn=None):
        """
        restarts the experiment with the same parameters
        """
        ntr_fn = new_tr_fn or self.new_trial
        stp_fn = step_fn or self.step
        obs = self.env.reset(new_tr_fn=ntr_fn, step_fn=stp_fn)
        return obs

    def step(self, action, new_tr_fn=None):
        """Public interface for the environment."""
        # TODO: Relying on private interface will break some gym behavior
        ntr_fn = new_tr_fn or self.new_trial
        obs, reward, done, info = self.env.step(action, new_tr_fn=ntr_fn)
        return obs, reward, done, info

```



