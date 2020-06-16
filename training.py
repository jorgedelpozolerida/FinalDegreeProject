"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import os
import matplotlib.pyplot as plt
import nalt_perceptualdecisionmaking as nalt
import variable_nch as vn
import trial_hist as th
import helper_functions as hf
from neurogym.wrappers import monitor, pass_action, pass_reward

from stable_baselines.common.policies import LstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import ACER  # , PPO2, ACKTR, A2C


def train(main_folder, timings, performance=True, l_n_ch=[2], instances=1,
          total_timesteps=1000000000, l_prob_2=[None], l_prob=[0.95],
          l_blocks=[[100, 200]], algs=[ACER], algs_names=['ACER'],
          perfect_integrator=True, window=50000, sv_per=50000,
          choice_transition_probs=True):

    for n_instance in range(instances):
        for blocks in l_blocks:
            for ind_alg, alg in enumerate(algs):
                for prob_2 in l_prob_2:
                    for prob in l_prob:
                        for n_ch in l_n_ch:
                            save_folder = \
                                main_folder + '/' + 'instance_' +\
                                str(n_instance+1) + '_' + str(blocks[0]) +\
                                '_' + str(blocks[1]) + '_nch_' + str(n_ch) +\
                                '_prob_' + str(prob) \
                                + '_algorithm_' + algs_names[ind_alg] \
                                + '_prob2_' + str(prob_2) + '/'
                            if not os.path.exists(save_folder +
                                                  'bhvr_data_all.npz'):

                                # ENVIRONMENT

                                env = nalt.nalt_PerceptualDecisionMaking(
                                    timing=timings, n_ch=n_ch, ob_nch=True,
                                    ob_histblock=False)
                                env = th.TrialHistory(env, block_dur=blocks[1],
                                                      probs=prob)
                                if blocks[0] > 0:
                                    env = vn.Variable_nch(env, block_nch=blocks[0],
                                                          prob_2=prob_2)
                                env = pass_action.PassAction(env)
                                env = pass_reward.PassReward(env)
                                env = monitor.Monitor(
                                    env, folder=save_folder, sv_per=sv_per,
                                    sv_fig=True, verbose=True, fig_type='svg')

                                # TRAINING

                                env = DummyVecEnv([lambda: env])
                                model = alg(
                                    LstmPolicy, env, verbose=0,
                                    policy_kwargs={'feature_extraction': "mlp"})
                                model.learn(total_timesteps=total_timesteps)

                                # PLOTTING

                                if performance:
                                    # Plot performance with perfect integrator.
                                    hf.plot_performance(
                                        folder=save_folder, window=window,
                                        perfect_integrator=perfect_integrator)
                                    plt.close('all')

                                if choice_transition_probs:
                                    hf.plot_transition_probs(save_folder)
                                    # Plot choice transition probs.
                                    hf.plot_choice_transition_probs(
                                        folder=save_folder,
                                        after_correct=True,
                                        col_bar='specific')
                                    hf.plot_choice_transition_probs(
                                        folder=save_folder,
                                        after_correct=False,
                                        col_bar='specific')
                                    plt.close('all')
                            else:
                                print('DONE')


if __name__ == '__main__':

    # Where to save my Results
    instances = 2
    main_folder = '/home/molano/multiple_choice/tests'

    # Combination of parameters for the model
    timings = {
            'fixation': ('constant', 200),
            'stimulus': ('constant', 300),
            'decision': ('constant', 200)}
    l_n_ch = [4]  # , 8 , 10]
    l_prob_2 = [None]
    l_prob = [0.95]  # , 0.99]
    l_blocks = [[500, 200]]  # [[100,200],[1000,200]] #, [200,400]]

    algs = [ACER]  # ,A2C, PPO2, ACKTR]
    algs_names = ['ACER']  # , 'A2C', 'PPO2', 'ACKTR']

    # Saving and Plotting parameters
    sv_per = 10000
    total_timesteps = 1000000
    window = 10000
    performance = True  # If performance plotted.
    perfect_integrator = True  # If perfect integrator performance plotted.
    choice_transition_probs = True  # If choice transition probs plottes.

    train(main_folder, timings, performance, l_n_ch, instances,
          total_timesteps, l_prob_2, l_prob, l_blocks, algs, algs_names,
          perfect_integrator, window, sv_per, choice_transition_probs)
