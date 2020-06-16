"""
Created on March 2020

@author: jorgedelpozolerida


Plots for all possible combinations between environment and wrappers.

There are 7 possible scenarios:

0. PerceptualDecisionMaking Default.
1. PerceptualDecisionMaking - General functioning.
2. PerceptualDecisionMakingDelayResponse.
3. PerceptualDecisionMaking + trial_hist
4. Nalt_PerceptualDecisionMaking.
5. Nalt_PerceptualDecisionMaking + trial_hist
6. Nalt_PerceptualDecisionMaking + n_variable
7. Nalt_PerceptualDecisionMaking + n_variable + trial_hist


To select which to plot just change "plot_num" value to the desired one.

To change conditions for each case change variables relative to that case.

"""

import sys
import os
import nalt_perceptualdecisionmaking as nalt
import n_alt_variable as nav
import trial_hist as th
import neurogym.utils.plotting as pl
import gym
import neurogym as ngym
from neurogym.envs import perceptualdecisionmaking
import matplotlib.pyplot as plt
sys.path.append(os.path.expanduser("~/gym"))

# Lists of all environment and plotting variables. l_variable[i] contains
#  variable info for scenario i.

l_env = []
l_num_steps = []
l_num_trials = []
l_def_act = []
l_model = []
l_name = []
l_legend = []
l_ob_traces = []
l_fig_kwargs = []

# Default plotting arguments.

env0 = perceptualdecisionmaking.PerceptualDecisionMaking()
l_env.append(env0)
num_steps0 = 200
l_num_steps.append(num_steps0)
num_trials0 = None
l_num_trials.append(num_trials0)
def_act0 = None
l_def_act.append(def_act0)
model0 = None
l_model.append(model0)
name0 = None
l_name.append(name0)
legend0 = True
l_legend.append(legend0)
ob_traces0 = []
l_ob_traces.append(ob_traces0)
fig_kwargs0 = {}
l_fig_kwargs.append(fig_kwargs0)


# 1. PerceptualDecisionMaking - General functioning

env1 = perceptualdecisionmaking.PerceptualDecisionMaking(
    dt=25, timing={'stimulus': ('constant', 300)})
l_env.append(env1)
num_steps1 = 200
l_num_steps.append(num_steps1)
num_trials1 = 4
l_num_trials.append(num_trials1)
def_act1 = 0
l_def_act.append(def_act1)
model1 = None
l_model.append(model1)
name1 = None
l_name.append(name1)
legend1 = True
l_legend.append(legend1)
ob_traces1 = ['Fixation cue', 'Stimulus 1', 'Stimulus 2']
l_ob_traces.append(ob_traces1)
fig_kwargs1 = {}
l_fig_kwargs.append(fig_kwargs1)


# 2. PerceptualDecisionMakingDelayResponse

env2 = perceptualdecisionmaking.PerceptualDecisionMakingDelayResponse()
l_env.append(env2)
num_steps2 = 200
l_num_steps.append(num_steps2)
num_trials2 = None
l_num_trials.append(num_trials2)
def_act2 = None
l_def_act.append(def_act2)
model2 = None
l_model.append(model2)
name2 = None
l_name.append(name2)
legend2 = True
l_legend.append(legend2)
ob_traces2 = []
l_ob_traces.append(ob_traces2)
fig_kwargs2 = {}
l_fig_kwargs.append(fig_kwargs2)


# 3. PerceptualDecisionMaking + trial_hist

prob = 0.9
block_dur = 30
env3 = nalt.nalt_PerceptualDecisionMaking(n_ch=2, dt=100, timing={'stimulus':
                                          ('constant', 500)},
                                          show_curr_block=True)
env3 = th.TrialHistory(env3, prob=prob, block_dur=block_dur)
l_env.append(env3)
num_steps3 = 160
l_num_steps.append(num_steps3)
num_trials3 = 180
l_num_trials.append(num_trials3)
def_act3 = 1
l_def_act.append(def_act3)
model3 = None
l_model.append(model3)
name3 = 'PerceptualDecisionMaking with Trial History Bias'
l_name.append(name3)
legend3 = True
l_legend.append(legend3)
ob_traces3 = []
l_ob_traces.append(ob_traces3)
fig_kwargs3 = {}
l_fig_kwargs.append(fig_kwargs3)


# 4. Nalt_PerceptualDecisionMaking

timings = {
        'fixation': ('constant', 200),
        'stimulus': ('constant', 500),
        'decision': ('constant', 200)}
show_curr_block = False
n_ch_in_ob = True
n_ch = 4

env4 = nalt.nalt_PerceptualDecisionMaking(dt=50, n_ch=n_ch, timing=timings,
                                          show_curr_block=show_curr_block,
                                          n_ch_in_ob=n_ch_in_ob)
l_env.append(env4)
num_steps4 = 200
l_num_steps.append(num_steps4)
num_trials4 = 5
l_num_trials.append(num_trials4)
def_act4 = None
l_def_act.append(def_act4)
model4 = None
l_model.append(model4)
name4 = '{:d}-Alternative '.format(n_ch)
l_name.append(name4)
legend4 = True
l_legend.append(legend4)
ob_traces4 = ['Fixation Cue']
ob_traces4.extend([('Stimulus ' + str(i)) for i in range(1, n_ch+1)])
if show_curr_block:
    ob_traces4.append('Current Block')
if n_ch_in_ob:
    ob_traces4.append('Active choices')
l_ob_traces.append(ob_traces4)
fig_kwargs4 = {}
l_fig_kwargs.append(fig_kwargs4)


# 5.Nalt_PerceptualDecisionMaking + trial_hist

timings = {
        'fixation': ('constant', 200),
        'stimulus': ('constant', 500),
        'decision': ('constant', 200)}
show_curr_block = True
n_ch_in_ob = False
n_ch = 4
prob = 0.9  # High to better show block type.
block_dur = 15

env5 = nalt.nalt_PerceptualDecisionMaking(dt=50, n_ch=n_ch, timing=timings,
                                          show_curr_block=show_curr_block,
                                          n_ch_in_ob=n_ch_in_ob)
env5 = th.TrialHistory(env5, prob=prob, block_dur=block_dur)
l_env.append(env5)
num_steps5 = 200
l_num_steps.append(num_steps5)
num_trials5 = 60
l_num_trials.append(num_trials5)
def_act5 = 1  # To make groundtruth more visible
l_def_act.append(def_act5)
model5 = None
l_model.append(model5)
name5 = '{:d}-Alternative with Trial History Bias'.format(n_ch)
l_name.append(name5)
legend5 = True
l_legend.append(legend5)
ob_traces5 = ['Fixation Cue']
ob_traces5.extend([('Stimulus ' + str(i)) for i in range(1, n_ch+1)])
if show_curr_block:
    ob_traces5.append(' 1: Alternating, 2: Repeating')
if n_ch_in_ob:
    ob_traces5.append('Active choices')
l_ob_traces.append(ob_traces5)
fig_kwargs5 = {}
l_fig_kwargs.append(fig_kwargs5)


# 6. Nalt_PerceptualDecisionMaking + n_variable

timings = {
        'fixation': ('constant', 200),
        'stimulus': ('constant', 500),
        'decision': ('constant', 200)}
show_curr_block = False
n_ch_in_ob = True
n_ch = 5
block_dur = 15


env6 = nalt.nalt_PerceptualDecisionMaking(dt=50, n_ch=n_ch, timing=timings,
                                          show_curr_block=show_curr_block,
                                          n_ch_in_ob=n_ch_in_ob)
env6 = nav.n_variable(env6, block_dur=block_dur)
l_env.append(env6)
num_steps6 = 200
l_num_steps.append(num_steps6)
num_trials6 = 75
l_num_trials.append(num_trials6)
def_act6 = 1  # To better see blocks and groundtruth
l_def_act.append(def_act6)
model6 = None
l_model.append(model6)
name6 = '[1,{:d}]-Alternative with Trial History Bias'.format(n_ch)
l_name.append(name6)
legend6 = True
l_legend.append(legend6)
ob_traces6 = ['Fixation Cue']
ob_traces6.extend([('Stimulus ' + str(i)) for i in range(1, n_ch+1)])
if show_curr_block:
    ob_traces6.append(' 1: Alternating, 2: Repeating')
if n_ch_in_ob:
    ob_traces6.append('Active choices')
l_ob_traces.append(ob_traces6)
fig_kwargs6 = {}
l_fig_kwargs.append(fig_kwargs6)


# 7. Nalt_PerceptualDecisionMaking + n_variable + trial_hist
timings = {
        'fixation': ('constant', 200),
        'stimulus': ('constant', 500),
        'decision': ('constant', 200)}
show_curr_block = True
n_ch_in_ob = True
n_ch = 3
prob = 0.95  # High to better show block type.
block_nch = 100
block_tr = 200


env7 = nalt.nalt_PerceptualDecisionMaking(dt=50, n_ch=n_ch, timing=timings,
                                          show_curr_block=show_curr_block,
                                          n_ch_in_ob=n_ch_in_ob)
env7 = th.TrialHistory(env7, prob=prob, block_dur=block_tr)
env7 = nav.n_variable(env7, block_dur=block_nch)
l_env.append(env7)
num_steps7 = 200
l_num_steps.append(num_steps7)
num_trials7 = 3000
l_num_trials.append(num_trials7)
def_act7 = 1  # To better see groundtruh
l_def_act.append(def_act7)
model7 = None
l_model.append(model7)
name7 = 'Variable [1,{:d}]-Alternative with Trial History Bias'.format(n_ch)
l_name.append(name7)
legend7 = True
l_legend.append(legend7)
ob_traces7 = ['Fixation Cue']
ob_traces7.extend([('Stimulus ' + str(i)) for i in range(1, n_ch+1)])
if show_curr_block:
    ob_traces7.append(' 1: Alternating, 2: Repeating')
if n_ch_in_ob:
    ob_traces7.append('Active choices')
l_ob_traces.append(ob_traces7)
fig_kwargs7 = {}
l_fig_kwargs.append(fig_kwargs7)


# CHOOSE WHICH SCENARIO TO PLOT BY CHANGING "plot_num".

plot_num = 1  # Which case to plot (1/2/3/....n)

env = l_env[plot_num]
num_steps = l_num_steps[plot_num]
num_trials = l_num_trials[plot_num]
def_act = l_def_act[plot_num]
model = l_model[plot_num]
name = l_name[plot_num]
legend = l_legend[plot_num]
ob_traces = l_ob_traces[plot_num]
fig_kwargs = l_fig_kwargs[plot_num]


ngym.utils.plot_env(env, num_steps=num_steps, num_trials=num_trials,
                    def_act=def_act, model=model, name=name, legend=legend,
                    ob_traces=ob_traces, fig_kwargs=fig_kwargs)
plt.show()
