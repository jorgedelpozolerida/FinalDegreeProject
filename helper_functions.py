#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
/*
 * @Author: jorgedelpozolerida
 * @Date: 2020-04-07 18:39:43
 * @Last Modified by: jorgedelpozolerida
 * @Last Modified time: 2020-06-01 11:54:04
 */

"""
import numpy as np
from numpy import logical_and as and_
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.optimize import curve_fit
from scipy.stats import entropy
import plotting_functions as pf
import get_activity as ga
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import ntpath
import glob
import sys
sys.path.append(os.path.expanduser("~/neurogym"))
import neurogym.utils.plotting as pl
rojo = np.array((228, 26, 28))/255
azul = np.array((55, 126, 184))/255
verde = np.array((77, 175, 74))/255
morado = np.array((152, 78, 163))/255
naranja = np.array((255, 127, 0))/255
marron = np.array((166, 86, 40))/255
amarillo = np.array((155, 155, 51))/255
rosa = np.array((247, 129, 191))/255
cyan = np.array((0, 1, 1))
gris = np.array((.5, .5, 0.5))
azul_2 = np.array([56, 108, 176])/255
rojo_2 = np.array([240, 2, 127])/255

COLORES = np.concatenate((azul.reshape((1, 3)), rojo.reshape((1, 3)),
                          verde.reshape((1, 3)), morado.reshape((1, 3)),
                          naranja.reshape((1, 3)), marron.reshape((1, 3)),
                          amarillo.reshape((1, 3)), rosa.reshape((1, 3))),
                         axis=0)
model_cols = ['evidence',
              'L+1', 'L-1', 'L+2', 'L-2', 'L+3', 'L-3', 'L+4', 'L-4',
              'L+5', 'L-5', 'L+6-10', 'L-6-10',
              'T++1', 'T+-1', 'T-+1', 'T--1', 'T++2', 'T+-2', 'T-+2',
              'T--2', 'T++3', 'T+-3', 'T-+3', 'T--3', 'T++4', 'T+-4',
              'T-+4', 'T--4', 'T++5', 'T+-5', 'T-+5', 'T--5',
              'T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10', 'intercept']
afterc_cols = [x for x in model_cols if x not in ['L+2', 'L-1', 'L-2',
                                                  'T+-1', 'T--1']]
aftere_cols = [x for x in model_cols if x not in ['L+1', 'T++1',
                                                  'T-+1', 'L+2',
                                                  'L-2']]


# SECONDARY FUNCTIONS

def get_tag(tag, file):
    '''
    Gets value of tag given a file.
    '''
    # process name
    file_name = ntpath.basename(file)
    start_val = file_name.find(tag)
    assert start_val != -1, 'Tag ' + tag + ' not found in ' + file_name
    val = file_name[start_val + len(tag) + 1:]
    val = val[:val.find('_')] if '_' in val else val
    return val


def evidence_mask(ev, percentage=10):
    '''
    Selects trials where stimulus is lower than percetile 'percentage'
    '''
    ev_abs = np.abs(ev)
    return ev_abs < np.percentile(ev_abs, percentage)


def get_times(num, per, step):
    '''
    Creates list of steps or times.
    Num is maximum number of steps available, per the top limit and step
    the spacing between each step.
    '''
    if per >= num:
        times = np.array([0])
    else:
        times = np.linspace(0, num - per, (num - per)//step + 1, dtype=int)
    return times


def get_repetitions(mat):
    """
    Makes diff of the input vector, mat, to obtain the repetition vector X,
    i.e. X will be 1 at t if the value of mat at t is equal to that at t-1
    """
    mat = mat.flatten()
    values = np.unique(mat)
    # We need to account for size reduction of np.diff()
    rand_ch = np.array(np.random.choice(values, size=(1,)))
    repeat_choice = np.concatenate((rand_ch, mat))
    return (np.diff(repeat_choice) == 0)*1


def probit(x, beta, alpha):
    '''
    Probit function with parameters alpha and beta.
    '''
    probit = 1/2*(1+erf((beta*x+alpha)/np.sqrt(2)))
    return probit


def probit_lapse_rates(x, beta, alpha, piL, piR):
    """
    Builds probit function (inverse of the cumulative distribution
    function of the standard normal distribution).
    If piR/piL are not zero, it will provide the lapse
    rate probit fit.
    """
    piR = 0
    piL = 0
    probit_lr = piR + (1 - piL - piR) * probit(x, beta, alpha)
    return probit_lr


def remove_borders(mask):
    '''
    Refines mask by removing blocks' borders, which are detected by
    a change of 1 or -1 (from True to False or viceversa).
    '''
    mask = 1*mask
    inside_blk_indx_on = np.diff(mask) != 1
    inside_blk_indx_on = np.append(False, inside_blk_indx_on)
    inside_blk_indx_off = np.diff(mask) != -1
    inside_blk_indx_off = np.append(inside_blk_indx_off, False)
    mask = and_.reduce((inside_blk_indx_on, inside_blk_indx_off, mask))
    return mask


def template_match(mat, templ):
    temp_match = np.convolve(mat, np.flip(templ), mode='same')
    times = (np.where(temp_match == np.max(temp_match))[0] +
             np.ceil(len(templ)/2)-1).astype('int')
    mask = np.zeros_like(mat)
    times = times[times < mask.shape[0]]
    mask[times] = 1
    return mask


def get_average_mat_cond(mat_cond):
    '''
    Averages across instances data contained in mat_cond
    into account its shape and data types. If training time is not
    equal across instances, average is calculated in sections or tranches.
    '''
    # Remove empty instances from data
    a_mat_cond = [x for x in mat_cond if len(x) > 0]
    max_ = np.max([len(x) for x in a_mat_cond])
    # a_mat_cond_ = [x[:min_] for x in a_mat_cond]
    a_mat_cond_ =\
        [np.concatenate((x, np.nan*np.ones((((int(max_-len(x)),)+np.array(x).shape[1:])))))
         for x in a_mat_cond]  # add nan to have same shape mats
    a_mat_cond_ = np.array(a_mat_cond_)
    average_matrix = np.nanmean(a_mat_cond_, axis=0)
    return average_matrix


# PRIMARY FUNCTIONS

def compute_transition_probs_mat(data_mat, choices, block_n_ch,
                                 block_tr_hist,
                                 extra_condition=None):
    '''
    Computes transition probs mat.
    '''
    if extra_condition is None:
        extra_condition = np.full(data_mat.shape, True, dtype=None)

    # get transition blocks
    blck_tr_hist_id = np.unique(block_tr_hist)
    n_blcks_trh = blck_tr_hist_id.shape[0]
    # get number of choices blocks
    blck_n_ch_id = np.unique(block_n_ch)
    n_blcks_nch = blck_n_ch_id.shape[0]

    # get choices
    ch_bins = np.append(choices-0.5, choices[-1]+0.5)
    trans_mat = np.empty((n_blcks_trh, n_blcks_nch, choices.shape[0],
                          choices.shape[0]))
    counts_mat = np.empty((n_blcks_trh, n_blcks_nch, choices.shape[0],
                           choices.shape[0]))
    for ind_nch, bl_nch in enumerate(blck_n_ch_id):
        for ind_trh, bl_trh in enumerate(blck_tr_hist_id):
            for ind_ch, ch in enumerate(choices):
                # avoid blocks borders
                blk_nch_mask = block_n_ch == bl_nch
                blk_nch_mask = remove_borders(blk_nch_mask)
                blk_trh_mask = block_tr_hist == bl_trh
                blk_trh_mask = remove_borders(blk_trh_mask)
                condition = and_.reduce((data_mat == ch, blk_nch_mask,
                                         blk_trh_mask, extra_condition))
                indx = np.where(condition)[0]
                indx = indx[indx < len(data_mat)-1]
                next_ch = data_mat[indx + 1]
                counts = np.histogram(next_ch, ch_bins)[0]
                trans_mat[ind_trh, ind_nch, ind_ch, :] =\
                    counts/np.sum(counts)  # /np.sum(counts)
                counts_mat[ind_trh, ind_nch, ind_ch, :] =\
                    counts.astype(int)
    print(np.sum(counts_mat))
    return trans_mat, counts_mat


def count_choices(folder, after_correct=False,
                  col_bar='general', sim_agent=False,
                  prop_wrong=None, period=None,
                  plot_figs=True, show_values=False,
                  sv_fig=True):
    """
    Plots choice-transition probabilities once the agent has trained in
    an environment for those trials at the end of each block where there
    is no obvious higher stimulus and after a correct previous choice.
    (criteria: v=max(accumulated_stimuli) < percentile_10(v)).
    Casuistry: end of nch block (vertical axis) against end of repetition
    block (horizontal axis).
    Inside each subplot, for current choice (row) at trial t, probability
    of the other choices at trial t+1.
    Figure is saved in folder.

    Args:
        folder: path, folder from where to extarct data and save results.
        after_correct: bool, if True,  choice transition probabilities
        for trials after a correct trial are plotted.
        col_bar: str, determines whether to plot colorbar and how. If
        'general', colobar common for all plots is added. If 'specific',
        colorbar for each subplot is added. If None (default), no colorbar.
        sim_agent: if True, a simulated agent is generated
        prop_wrong: if stim_agent==True, proportion of wrong choices for it
        period: limits number of steps selected for calculation of bias
        plot_figs: if True, figures are plotted
        show_values: if True, total counts for each condition shown on
        top of normal plot
        sv_fig: if True, figure is saved in folder.

    """
    # TODO: split creation of matrix from plotting of matrix
    # trans_mat, counts_mat, choices, col_bar, n_blcks_nch,
    # n_blcks_trh, blck_tr_hist_id, blck_n_ch_id, show values,
    # (title, period: for suptitle)

    # Data loading/Putting together and loading.
    if os.path.exists(folder + '/bhvr_data_all.npz'):
        data = np.load(folder + '/bhvr_data_all.npz')
    else:
        data = pl.put_together_files(folder)

    if data:
        print('Number of trials: ', str(data['gt'].shape[0]))
        if period is None:
            period = [0, data['gt'].shape[0]]
        elif isinstance(period, int):
            period = [data['gt'].shape[0]-period, data['gt'].shape[0]]
        print(period)
        # We select necessary data
        choices_agent = data['choice'][period[0]:period[1]]
        print('CHOICES: \n', np.unique(choices_agent, return_counts=True)[1][-2])


def process_cond(main_folder, cond, val, step=30000, acr_tr_per=300000,
                 plot_figs=True, test_retrain=''):
    '''
    Generates a dictionary  of data for cond in main_folder containing:
            'val_mat', 'trans_mats_ac', 'shapes',
            'trans_mats_ae', 'bias_mats_psych',
            'bias_mats_entr', 'perf_mats',
            'glm_mats_ac', 'glm_mats_ae'
    and saves it in  .npz format in same folder.
    '''
    files = glob.glob(main_folder+'/alg*'+cond+'*')
    val_mat = []
    trans_mats_ac = []
    trans_mats_ae = []
    bias_mats_entr = []
    bias_mats_psych = []
    glm_mats_ac = []
    glm_mats_ae = []
    perf_mats = []
    shapes = []
    bias_rebound = []
    bias_sequence = []
    for ind_f, f in enumerate(files):
        print('-------------------------')
        print(f)
        # Get values from each file.
        curr_val = get_tag(val, f)
        work_f = f + '/' + test_retrain
        trans_ac, trans_ae, weights_ac, weights_ae, bias_psych, bias_entr, perf,\
            bias_bef_aft, bias_seq = process_data(work_f, acr_tr_per=acr_tr_per,
                                                  step=step, trans_mats=True,
                                                  glm=True, bias=True,
                                                  plot_figs=plot_figs)
        bias_rebound.append(bias_bef_aft)
        bias_sequence.append(bias_seq)
        bias_mats_psych.append(bias_psych)
        bias_mats_entr.append(bias_entr)
        perf_mats.append(perf)
        glm_mats_ac.append(weights_ac)
        glm_mats_ae.append(weights_ae)
        # Append or add values to lists of data
        val_mat.append(curr_val)
        if trans_ac is not None:
            trans_mats_ac.append(trans_ac.flatten())  # shape + flattened
            trans_mats_ae.append(trans_ae.flatten())
            shapes.append(trans_ac.shape)
        else:
            trans_mats_ac.append([])  # shape + flattened
            trans_mats_ae.append([])
            shapes.append(0)
        plt.close('all')

    data = {'val_mat': val_mat, 'trans_mats_ac': trans_mats_ac, 'shapes': shapes,
            'trans_mats_ae': trans_mats_ae, 'bias_mats_psych': bias_mats_psych,
            'bias_mats_entr': bias_mats_entr, 'perf_mats': perf_mats,
            'glm_mats_ac': glm_mats_ac, 'glm_mats_ae': glm_mats_ae,
            'bias_rebound': bias_rebound, 'bias_sequence': bias_sequence,
            'step': step, 'acr_tr_per': acr_tr_per}

    np.savez(main_folder + '/data_' + cond + test_retrain + '.npz', **data)

    return data


def test_trained_networks(main_folder, cond, val, num_steps=1000000, sv_per=10000,
                          seed=0, test_retrain='test'):
    '''
    Generates a dictionary  of data for cond in main_folder containing:
            'val_mat', 'trans_mats_ac', 'shapes',
            'trans_mats_ae', 'bias_mats_psych',
            'bias_mats_entr', 'perf_mats',
            'glm_mats_ac', 'glm_mats_ae'
    and saves it in  .npz format in same folder.
    '''
    files = glob.glob(main_folder+'/alg*'+cond+'*')
    for ind_f, f in enumerate(files):
        sv_folder = f + '/' + test_retrain+'/'
        ga.get_activity(folder=f, alg=cond, sv_folder=sv_folder, seed=seed,
                        num_steps=num_steps, test_retrain=test_retrain,
                        sv_per=sv_per)


def batch_analysis(main_folder, conds=['A2C', 'ACER', 'ACKTR', 'PPO2'],
                   val='n_ch', step=10000, acr_tr_per=300000, n_steps_test=100000,
                   analyze_tr_data=True, test_retrain='test', sv_per_test=10000):
    '''
    Saves data for each given condition in single .npz file, which contains
    a dict of concatenated relevant data separated according to val.
    '''
    for cond in conds:
        if analyze_tr_data:
            process_cond(main_folder, cond, val, step=step, acr_tr_per=acr_tr_per)
        if test_retrain != '':
            test_trained_networks(main_folder, cond, val, num_steps=n_steps_test,
                                  test_retrain=test_retrain, sv_per=sv_per_test)
            process_cond(main_folder, cond, val, step=None, acr_tr_per=None,
                         test_retrain=test_retrain, plot_figs=False)


def bias_calculation(choice, ev, mask, bias_type='entropy', num_chs=None):
    """
    Computes repeating bias given the choice of the netwnum_chsork, the
    stimulus evidence and a mask indicating the trials on which the bias
    should be computed.
    """

    if bias_type == 'entropy':
        # Associate invalid trials (network fixates) with incorrect choice.
        invalid = choice == 0
        num_invalids = np.sum(invalid)

        if num_chs is None:
            num_chs = np.unique(choice).shape[0] - 1*(num_invalids > 0)
        # assign random choices to the invalid trials
        aux = np.random.choice(num_chs, (num_invalids,)) + 1
        choice[invalid] = aux
        # one entropy value calculated for trial t + 1 after choice ch
        entropies = []
        for ch in np.arange(1, num_chs+1):
            inds_ch = choice[:-1] == ch
            # t + 1 selected where choice at t == ch
            inds_ch = np.concatenate((np.array([0]), inds_ch))
            inds_ch = np.where(and_(inds_ch, mask))[0]
            _, counts = np.unique(choice[inds_ch], return_counts=True)
            counts = counts[:num_chs]
            # calculation: (1+sum(pk * log(pk)))/max_entropy
            prob = counts / np.sum(counts)
            entropies.append(1-entropy(prob)/np.log(num_chs))
        popt = [np.nan, np.mean(entropies)]
        pcov = 0
    elif bias_type == 'simple_bias':
        choice[choice == 0] = ev[choice == 0] > 0
        repeat = get_repetitions(choice)
        # choice_repeating is just the original right_choice mat
        # but shifted one element to the left.
        choice_repeating = np.concatenate(
            (np.array(np.random.choice([0, 1])).reshape(1, ),
             choice[:-1]))
        # the rep. evidence is the original evidence with a negative sign
        # if the repeating side is the left one
        rep_ev = ev *\
            (-1)**(choice_repeating == 2)
        rep_ev_mask = rep_ev[mask]  # xdata
        repeat_mask = repeat[mask]  # ydata
        try:
            # Use non-linear least squares to fit probit to xdata, ydata
            popt, pcov = curve_fit(probit_lapse_rates, rep_ev_mask,
                                   repeat_mask, maxfev=10000)
            # popt: optimal values for the parameters so that the sum of the
            #  squared residuals of f(xdata, *popt) - ydata is minimized

        except RuntimeError:
            popt = [np.nan, np.nan]
            pcov = 0
            print('no fitting')
    return popt, pcov


def compute_AE_bias_and_rebound(ch, ev, perf, nch_mask=None, tr_hist_mask=None,
                                bias_type='simple_bias', ev_perc=10, nch=None,
                                template_perf=None, template_rep=None,
                                templ_specs=None, whole_seq=True):
    '''
    Computes bias for passed conditions. For each block type in tr_block,
    after_error and after_correct. Shape = (2, num_blocks)

    Args

    ch: choices array
    ev: evidence array
    perf: performance array
    tr_block: transition block array
    c_tr: catch trials block array
    nch_mask: mask selecting only nch values
    nch: active choices block for which to compute bias
    figs: bool, whether to plot figures for above arguments or not
    new_fig: False
    lw: line width of plot if figs = True
    comp_bias: bool, whether to compute bias or not
    bias_type: 'entropy' or 'simple_bias'
    ev_perc: percetile of evidence used for bias calculation

    '''
    templ_specs_def = {'num_trials_seq': 3, 'after_corr': 0, 'rep_alt': 1}
    repetitions = get_repetitions(ch)
    mean_perf = np.mean(perf)
    mean_rep = np.mean(repetitions)
    perf_zero_mean = perf - mean_perf
    rep_zero_mean = repetitions - mean_rep
    if bias_type == 'entropy':
        # select evidence below ev_perc quantile
        ev_mask = evidence_mask(ev, percentage=ev_perc)
        # get rid of first value in evidence
        ev_mask = np.concatenate((ev_mask[1:], np.array([0])))
    else:
        ev_mask = np.ones_like(perf) == 1
    if nch_mask is None:
        nch_mask = np.ones_like(perf) == 1
    if tr_hist_mask is None:
        tr_hist_mask = np.ones_like(perf) == 1
    nch_mask = remove_borders(nch_mask)
    tr_hist_mask = remove_borders(tr_hist_mask)
    if template_perf is None:
        templ_specs_def.update(templ_specs)
        nt_seq = templ_specs_def['num_trials_seq']
        ac = templ_specs_def['after_corr']
        template_perf = np.array([-1, 1] + [1]*nt_seq + [ac] + [1])
    if template_rep is None:
        templ_specs_def.update(templ_specs)
        nt_seq = templ_specs_def['num_trials_seq']
        rep_alt = templ_specs_def['rep_alt']
        template_rep = np.array([rep_alt]*nt_seq)
    assert len(template_perf) == len(template_rep)
    if whole_seq:
        num_steps = len(template_perf)
    else:
        num_steps = 3
    biases = np.empty((num_steps, ))
    # TODO: if only the before/after error biases are needed this for loop could be
    # done shorter
    for count, ind_seq in enumerate(range(len(template_perf)-num_steps,
                                          len(template_perf))):
        ind_perf = ind_seq + 1
        ind_rep = ind_seq + 1
        templ_perf = template_perf[:max(0, ind_perf)]
        templ_rep = template_rep[:max(0, ind_rep)]
        perf_mask = template_match(perf_zero_mean, templ_perf)
        if len(templ_rep) > 0:
            rep_mask = template_match(rep_zero_mean, templ_rep)
        else:
            rep_mask = np.ones_like(perf_mask)
        templ_mask = and_(rep_mask, perf_mask)
        mask = and_.reduce((ev_mask, nch_mask, tr_hist_mask, templ_mask))
        mask = np.concatenate((np.array([False]), mask[:-1]))
        if False:
            f, ax = plt.subplots(nrows=1, ncols=1)
            start = max(0, int(np.where(mask)[0][0] - 10))
            num_tr_plt = 20
            plt.plot(perf_zero_mean[start:start+num_tr_plt],
                     '+-r', label='perf')
            plt.plot(rep_zero_mean[start:start+num_tr_plt],
                     '+-b', label='repets')
            plt.plot(mask[start:start+num_tr_plt]-1, '+-k', label='mask')
            # plt.plot(nch_mask[start:start+num_tr_plt]-2, '+-c', label='nch')
            # plt.plot(tr_hist_mask[start:start+num_tr_plt]-3,
            #          '+-m', label='trh')
            # plt.plot(templ_mask[start:start+num_tr_plt]-4,
            #          '+-k', label='templ')
            plt.legend()
            plt.title(str(ind_seq))
            # asdda
        if np.sum(mask) > 10:
            # print(np.sum(mask))
            # print(templ_perf)
            # print(templ_rep)
            # print('--------')
            popt, pcov = bias_calculation(ch.copy(), ev.copy(), mask.copy(),
                                          bias_type=bias_type, num_chs=nch)
        else:
            popt = [np.nan, np.nan]
        # we just care about second element of popt, which is the bias.
        biases[count] = popt[1]
    # asdasd
    return biases


def compute_bias_diff_blocks(ch, ev, perf, tr_block, c_tr, nch_mask=None,
                             nch=None, figs=False, new_fig=False, lw=0.5,
                             comp_bias=True, bias_type='entropy',
                             ev_perc=10):
    '''
    Computes bias for passed conditions. For each block type in tr_block,
    after_error and after_correct. Shape = (2, num_blocks)

    Args

    ch: choices array
    ev: evidence array
    perf: performance array
    tr_block: transition block array
    c_tr: catch trials block array
    nch_mask: mask selecting only nch values
    nch: active choices block for which to compute bias
    figs: bool, whether to plot figures for above arguments or not
    new_fig: False
    lw: line width of plot if figs = True
    comp_bias: bool, whether to compute bias or not
    bias_type: 'entropy' or 'simple_bias'
    ev_perc: percetile of evidence used for bias calculation

    '''

    blocks = np.unique(tr_block)
    num_blocks = blocks.shape[0]
    biases = np.empty((2, num_blocks))
    if bias_type == 'entropy':
        # select evidence below ev_perc quantile
        ev_mask = evidence_mask(ev, percentage=ev_perc)
        # get rid of first value in evidence
        ev_mask = np.concatenate((ev_mask[1:], np.array([0])))
    else:
        ev_mask = np.ones_like(perf) == 1
    if nch_mask is None:
        nch_mask = np.ones_like(perf) == 1
    nch_mask = remove_borders(nch_mask)
    if figs:
        if new_fig:
            plt.subplots(figsize=(8, 8))
        labels = ['ae blk 0:', 'ae blk 1:', 'ac blk 0:', 'ac blk 1:']
        counter = 0
    # After error and after correct bias for each block.
    for ind_perf in range(2):
        # print('Perf', str(ind_perf))
        for ind_blk, blk in enumerate(blocks):
            # print('Block', str(blk))
            tr_block_mask = tr_block == blk
            tr_block_mask = remove_borders(tr_block_mask)
            mask = and_.reduce((perf == ind_perf, c_tr == 0, ev_mask, nch_mask,
                                tr_block_mask))
            mask = np.concatenate((np.array([False]), mask[:-1]))
            if ind_perf == 1 and ind_blk == 1 and False:
                f, ax = plt.subplots(figsize=(8, 8))
                num = 500
                start = 60000
                ax.plot(ch[start:start+num], '-+', label='choice', lw=1)
                ax.plot(perf[start:start+num]-3, '--+', label='perf', lw=1)
                ax.plot(mask[start:start+num]-3, '-+', label='mask', lw=1)
                ax.plot(tr_block[start:start+num], '-+',
                        label='trans. blck', lw=1)
                ax.plot(c_tr[start:start+num], '-+', label='ctr', lw=1)
                # ax.plot(tr_block_mask[start:start+num], '--+',
                #         label='trans. blck mask', lw=1)
                ax.plot(nch_mask[start:start+num] - 4, '-+', label='nch', lw=1)
                ax.plot(ev[start:start+num], '-+', label='ev', lw=1)
                # ax.plot(ev_mask[start:start+num]+0.1, '-+',
                #         label='evMask', lw=1)
                for ind in range(num):
                    ax.plot([ind, ind], [-3, 3], '--', color=(.7, .7, .7))
                plt.legend()

            if np.sum(mask) > 100 and comp_bias:
                popt, pcov = bias_calculation(ch.copy(), ev.copy(),
                                              mask.copy(),
                                              bias_type=bias_type, num_chs=nch)
                if figs:
                    color = rojo if ind_blk == 0 else azul
                    x = np.linspace(np.min(ev),
                                    np.max(ev), 50)
                    label = labels[counter] + str(round(popt[1], 3))
                    alpha = 0.6+0.4*ind_perf
                    pf.plot_psycho_curve(x, popt, label, color, alpha, lw=lw)
                    counter += 1
            else:
                popt = [np.nan, np.nan]
            # we just care about second element of popt, which is the bias.
            biases[ind_perf, ind_blk] = popt[1]
    return biases


def load_behavioral_data(file):
    """
    Loads  file and returns relevant behavioural data from it.
    Specifically: choice, gt, performance, evidence, catch_trial,
    reward, stimulus, nch and tr_block.
    """
    data = np.load(file)
    choice = data['choice']
    stimulus = data['stimulus']
    if 'gt' in data.keys():
        gt = data['gt']
    else:
        gt = data['correct_side']
    evidence = stimulus[:, 1] - stimulus[:, 2]
    if 'catch_trial' in data.keys():
        catch_trial = data['catch_trial']
        if len(catch_trial) == 0:
            catch_trial = np.zeros((choice.shape[0],))
    else:
        catch_trial = np.zeros((choice.shape[0],))
    performance = choice == gt
    # assert (performance == data['performance']).all()
    if 'reward' in data.keys():
        reward = data['reward']
    else:
        reward = performance
    if 'nch' in data.keys():
        nch = data['nch']
    else:
        nch = np.ones_like((choice,))*2
    if 'curr_block' in data.keys():
        tr_block = data['curr_block']
    else:
        tr_block = np.zeros_like((choice,))
    data = {'choice': choice, 'gt': gt,
            'performance': performance, 'evidence': evidence,
            'catch_trial': catch_trial, 'reward': reward,
            'stimulus': stimulus, 'nch': nch, 'tr_block': tr_block}
    return data


def bias_across_training(data, per=100000, step=None,
                         cond_nch_blck=None, bias_type='entropy'):
    """
    Returns bias matrix and performance matrix for nch block passed.

    per: duration of window in which bias is compted
    step: None
    cond_nch_blck: nch block where bias is computed.
    bias_type: 'entropy'
    """
    choice = data['choice']
    evidence = data['evidence']
    performance = data['performance']
    catch_trials = data['catch_trial']
    if cond_nch_blck is not None:
        if cond_nch_blck > 1:
            nch_blck_mask = data['nch'] == cond_nch_blck
    else:
        # no mask actually
        nch_blck_mask = np.ones_like(data['choice']) == 1
    tr_block = data['tr_block']
    num_blocks = np.unique(tr_block).shape[0]
    if per is None:
        per = choice.shape[0]
    if step is None:
        step = per
    # Bias only computed for selected steps
    steps = get_times(choice.shape[0], per, step)
    bias_mat = np.empty((len(steps), 2, num_blocks))
    perf_mat = np.empty((len(steps)))
    for ind, ind_per in enumerate(steps):
        ev = evidence[ind_per:ind_per+per+1]
        perf = performance[ind_per:ind_per+per+1]
        ch = choice[ind_per:ind_per+per+1]
        c_tr = catch_trials[ind_per:ind_per+per+1]
        nch_mask = nch_blck_mask[ind_per:ind_per+per+1]
        tr_blk = tr_block[ind_per:ind_per+per+1]
        biases = compute_bias_diff_blocks(ch=ch, ev=ev, perf=perf, c_tr=c_tr,
                                          tr_block=tr_blk, nch_mask=nch_mask,
                                          nch=cond_nch_blck,
                                          bias_type=bias_type)
        perf_mat[ind] = np.mean(perf)
        # count the number of new trials (bias is computed with a sliding
        # window and different periods overlap)
        for ind_perf in range(2):
            for ind_blk in range(num_blocks):
                bias_mat[ind, ind_perf, ind_blk] = biases[ind_perf, ind_blk]

    return bias_mat, perf_mat


def get_bias_single_exp(folder, acr_tr_per=200000, step=40000,
                        bias_type='entropy', cond_nch_blck=None):
    '''
    Returns data_flag, data, bias_mat, perf_mat for data in folder.
    '''
    data_flag = pl.put_together_files(folder)
    if data_flag:
        files = glob.glob(folder + '/*bhvr_data_all*.npz')
        assert len(files) <= 1, 'More than one file matching specifications'
        file = files[0]
        data = load_behavioral_data(file)
        if acr_tr_per is None or data['choice'].shape[0] > acr_tr_per:
            bias_mat, perf_mat =\
                bias_across_training(data, per=acr_tr_per, step=step,
                                     bias_type=bias_type,
                                     cond_nch_blck=cond_nch_blck)
            return data_flag, data, bias_mat, perf_mat
        else:
            print('Not enough data')
            return False, [], [], []
    else:
        print('Could not put data together')
        return False, [], [], []


def get_GLM_regressors(data, cond_nch_blck=None):
    """
    input df object, since it will calculate history*, it must contain
    consecutive trials returns preprocessed dataframe.
    """
    ev_ = data['evidence']
    perf_ = data['performance'].astype('float')
    ch_ = data['choice'].astype('float')
    if cond_nch_blck is not None:  # make nan all choices larger than cond_nch_blck
        nan_indx = np.logical_or(ch_ > cond_nch_blck, ch_ == 0)
        ev_[nan_indx] = np.nan
        perf_[nan_indx] = np.nan
        ch_[nan_indx] = np.nan
    ch_ -= 1  # choices should belong to {0, 1}
    ch_ = 1*(ch_ == 0)
    # gt_ = data['gt']
    prev_perf_ = ~ np.concatenate((np.array([False]),
                                   data['performance'][:-1]))
    prev_perf_ = prev_perf_.astype('int')
    ev_ /= np.nanmax(ev_)
    rep_ch_ = get_repetitions(ch_)
    # variables:
    # 'origidx': trial index within session
    # 'rewside': ground truth
    # 'hithistory': performance
    # 'R_response': choice (right == 1, left == 0, invalid == nan)
    # 'subjid': subject
    # 'sessid': session
    # 'res_sound': stimulus (left - right) [frame_i, .., frame_i+n]
    # 'sound_len': stim duration
    # 'frames_listened'
    # 'aftererror': not(performance) shifted
    # 'rep_response'
    df = {'origidx': np.arange(ch_.shape[0]),
          'R_response': ch_,
          'hit': perf_,
          'evidence': ev_,
          'aftererror': prev_perf_,
          'rep_response': rep_ch_}
    df = pd.DataFrame(df)

    # Lateral module
    df['L+1'] = np.nan  # np.nan considering invalids as errors
    df.loc[(df.R_response == 1) & (df.hit == 1), 'L+1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 1), 'L+1'] = -1
    df.loc[df.hit == 0, 'L+1'] = 0
    df['L+1'] = df['L+1'].shift(1)
    df.loc[df.origidx == 1, 'L+1'] = np.nan
    # L-
    df['L-1'] = np.nan
    df.loc[(df.R_response == 1) & (df.hit == 0), 'L-1'] = 1
    df.loc[(df.R_response == 0) & (df.hit == 0), 'L-1'] = -1
    df.loc[df.hit == 1, 'L-1'] = 0
    df['L-1'] = df['L-1'].shift(1)
    df.loc[df.origidx == 1, 'L-1'] = np.nan
    # shifts
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df['L+'+str(item)] = df['L+'+str(item-1)].shift(1)
        df['L-'+str(item)] = df['L-'+str(item-1)].shift(1)
        df.loc[df.origidx == 1, 'L+'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'L-'+str(item)] = np.nan

    # add from 6 to 10, assign them and drop prev cols cols
    cols_lp = ['L+'+str(x) for x in range(6, 11)]
    cols_ln = ['L-'+str(x) for x in range(6, 11)]

    df['L+6-10'] = np.nansum(df[cols_lp].values, axis=1)
    df['L-6-10'] = np.nansum(df[cols_ln].values, axis=1)
    df.drop(cols_lp+cols_ln, axis=1, inplace=True)
    df.loc[df.origidx <= 6, 'L+6-10'] = np.nan
    df.loc[df.origidx <= 6, 'L-6-10'] = np.nan

    # pre transition module
    df.loc[df.origidx == 1, 'rep_response'] = np.nan
    df['rep_response_11'] = df.rep_response
    df.loc[df.rep_response == 0, 'rep_response_11'] = -1
    df.rep_response_11.fillna(value=0, inplace=True)
    df.loc[df.origidx == 1, 'aftererror'] = np.nan

    # transition module
    df['T++1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 1), 'T++1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 0), 'T++1'] = 0
    df['T++1'] = df['T++1'].shift(1)

    df['T+-1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 0) & (df.hit == 0), 'T+-1'] =\
        df.loc[(df.aftererror == 0) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 1) | (df.hit == 1), 'T+-1'] = 0
    df['T+-1'] = df['T+-1'].shift(1)

    df['T-+1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 1), 'T-+1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 1), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 0), 'T-+1'] = 0
    df['T-+1'] = df['T-+1'].shift(1)

    df['T--1'] = np.nan  # np.nan
    df.loc[(df.aftererror == 1) & (df.hit == 0), 'T--1'] =\
        df.loc[(df.aftererror == 1) & (df.hit == 0), 'rep_response_11']
    df.loc[(df.aftererror == 0) | (df.hit == 1), 'T--1'] = 0
    df['T--1'] = df['T--1'].shift(1)

    # shifts now
    for i, item in enumerate([2, 3, 4, 5, 6, 7, 8, 9, 10]):
        df['T++'+str(item)] = df['T++'+str(item-1)].shift(1)
        df['T+-'+str(item)] = df['T+-'+str(item-1)].shift(1)
        df['T-+'+str(item)] = df['T-+'+str(item-1)].shift(1)
        df['T--'+str(item)] = df['T--'+str(item-1)].shift(1)
        df.loc[df.origidx == 1, 'T++'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T+-'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T-+'+str(item)] = np.nan
        df.loc[df.origidx == 1, 'T--'+str(item)] = np.nan

    cols_tpp = ['T++'+str(x) for x in range(6, 11)]
    # cols_tpp = [x for x in df.columns if x.startswith('T++')]
    cols_tpn = ['T+-'+str(x) for x in range(6, 11)]
    # cols_tpn = [x for x in df.columns if x.startswith('T+-')]
    cols_tnp = ['T-+'+str(x) for x in range(6, 11)]
    # cols_tnp = [x for x in df.columns if x.startswith('T-+')]
    cols_tnn = ['T--'+str(x) for x in range(6, 11)]
    # cols_tnn = [x for x in df.columns if x.startswith('T--')]

    df['T++6-10'] = np.nansum(df[cols_tpp].values, axis=1)
    df['T+-6-10'] = np.nansum(df[cols_tpn].values, axis=1)
    df['T-+6-10'] = np.nansum(df[cols_tnp].values, axis=1)
    df['T--6-10'] = np.nansum(df[cols_tnn].values, axis=1)
    df.drop(cols_tpp+cols_tpn+cols_tnp+cols_tnn, axis=1, inplace=True)
    df.loc[df.origidx < 6, ['T++6-10', 'T+-6-10', 'T-+6-10', 'T--6-10']] =\
        np.nan
    # transforming transitions to left/right space
    for col in [x for x in df.columns if x.startswith('T')]:
        df[col] = df[col] * (df.R_response.shift(1)*2-1)
        # {0 = Left; 1 = Right, nan=invalid}

    df['intercept'] = 1
    df.loc[:, model_cols].fillna(value=0, inplace=True)

    return df  # resulting df with lateralized T+


def weights_across_training(data, per, step, cond_nch_blck):
    '''
    Returns weights of Logistic Regression split in after error and
    after correct cases, given data.
    '''
    nch_blk = data['nch']
    df = get_GLM_regressors(data, cond_nch_blck=cond_nch_blck)
    # condition on num-ch block
    if per is None:
        per = df.shape[0]
    steps = get_times(df.shape[0], per, step)
    weights_ac = []
    weights_ae = []
    for ind, ind_per in enumerate(steps):
        indx = and_(df.origidx.values >= ind_per,
                    df.origidx.values < ind_per+per)
        df_tmp = df[indx]
        if cond_nch_blck > 1:
            indx_nch = nch_blk[ind_per:ind_per+per] == cond_nch_blck
            df_tmp = df_tmp[indx_nch]
        not_nan_indx = df_tmp['R_response'].notna()
        X_df_ac, y_df_ac =\
            df_tmp.loc[(df_tmp.aftererror == 0) & not_nan_indx,
                       afterc_cols].fillna(value=0),\
            df_tmp.loc[(df_tmp.aftererror == 0) & not_nan_indx, 'R_response']
        X_df_ae, y_df_ae =\
            df_tmp.loc[(df_tmp.aftererror == 1) & not_nan_indx,
                       aftere_cols].fillna(value=0),\
            df_tmp.loc[(df_tmp.aftererror == 1) & not_nan_indx, 'R_response']

        if len(np.unique(y_df_ac.values)) == 2 and\
           len(np.unique(y_df_ae.values)) == 2:
            Lreg_ac = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                         solver='saga', random_state=123,
                                         max_iter=10000000, n_jobs=-1)
            Lreg_ac.fit(X_df_ac.values, y_df_ac.values)
            Lreg_ae = LogisticRegression(C=1, fit_intercept=False, penalty='l2',
                                         solver='saga', random_state=123,
                                         max_iter=10000000, n_jobs=-1)
            Lreg_ae.fit(X_df_ae.values, y_df_ae.values)
            weights_ac.append(Lreg_ac.coef_)
            weights_ae.append(Lreg_ae.coef_)
        else:
            weights_ac.append(np.ones((1, len(afterc_cols)))*np.nan)
            weights_ae.append(np.ones((1, len(aftere_cols)))*np.nan)
    weights_ac = np.asarray(weights_ac)
    weights_ae = np.asarray(weights_ae)

    return weights_ac, weights_ae


def get_glm_ws_single_exp(folder, acr_tr_per=200000, step=40000,
                          cond_nch_blck=0):
    data_flag = pl.put_together_files(folder)
    '''
    Extracts data from training files and calculates weights of GLM.
    '''
    if data_flag:
        files = glob.glob(folder + '/*bhvr_data_all*.npz')
        assert len(files) <= 1, 'More than one file matching specifications'
        file = files[0]
        data = load_behavioral_data(file)
        if acr_tr_per is None or data['choice'].shape[0] > acr_tr_per:
            weights_ac, weights_ae =\
                weights_across_training(data, per=acr_tr_per, step=step,
                                        cond_nch_blck=cond_nch_blck)
            return data_flag, data, weights_ac, weights_ae
        else:
            print('Not enough data')
            return False, [], [], []
    else:
        print('Could not put data together')
        return False, [], [], []


def get_rebound_single_exp(folder, min_num_tr=200000, period=-2000000,
                           cond_nch_blck=2):
    data_flag = pl.put_together_files(folder)
    '''
    Extracts data from training files and calculates weights of GLM.
    '''
    if data_flag:
        files = glob.glob(folder + '/*bhvr_data_all*.npz')
        assert len(files) <= 1, 'More than one file matching specifications'
        file = files[0]
        data = load_behavioral_data(file)
        if min_num_tr is None or data['choice'].shape[0] > min_num_tr:
            data = load_behavioral_data(file)
            choice = data['choice'][period:]
            evidence = data['evidence'][period:]
            performance = data['performance'][period:]
            nch_blck_mask = data['nch'][period:] == cond_nch_blck
            tr_hist_mask = None  # data['tr_block'][period:] == 1
            num_trials_seq = 6
            rep_alt = 1
            after_corr = -1
            biases = []
            for ind_n_tr_seq in range(1, num_trials_seq+1):
                whole_seq = ind_n_tr_seq == num_trials_seq
                template_rep = np.array([0]*2+[rep_alt]*ind_n_tr_seq+[0, 0])
                # after error
                template_perf =\
                    np.array([-1, 1] + [1]*ind_n_tr_seq + [after_corr] + [1])
                bss = compute_AE_bias_and_rebound(ch=choice, ev=evidence,
                                                  perf=performance,
                                                  nch_mask=nch_blck_mask,
                                                  tr_hist_mask=tr_hist_mask,
                                                  bias_type='simple_bias',
                                                  template_perf=template_perf,
                                                  template_rep=template_rep,
                                                  whole_seq=whole_seq)
                biases.append(np.array([bss[-3], bss[-1]]))
            return data_flag, biases, bss
        else:
            print('Not enough data')
            return False, [], []
    else:
        print('Could not put data together')
        return False, [], []


def process_data(file, acr_tr_per=300000, step=30000, nch=2, trans_mats=True,
                 glm=True, bias=True, rebound=True, plot_figs=True):
    if trans_mats:
        trans_ac = pf.plot_choice_transition_probs(file, after_correct=True,
                                                   period=acr_tr_per,
                                                   plot_figs=plot_figs,
                                                   show_values=True)
        trans_ae = pf.plot_choice_transition_probs(file, after_correct=False,
                                                   period=acr_tr_per,
                                                   plot_figs=plot_figs,
                                                   show_values=True)
    if glm:
        flag, _, weights_ac, weights_ae =\
            get_glm_ws_single_exp(file, acr_tr_per=acr_tr_per, step=step,
                                  cond_nch_blck=nch)
        if plot_figs and flag:
            tags_mat = [['evidence', 'intercept'], ['L+', 'L-'],
                        ['T++', 'T-+', 'T+-', 'T--']]
            figs = pf.plot_glm_weights(weights_ac, weights_ae,
                                       tags_mat, step, acr_tr_per)
            for ind_f, f in enumerate(figs):
                name = ''.join(tags_mat[ind_f])
                f.savefig(file+'/'+name+'.png', dpi=400,
                          bbox_inches='tight')
                plt.close(f)

    if bias:
        flag, _, bias_entr, perf =\
            get_bias_single_exp(file, acr_tr_per=acr_tr_per, step=step,
                                bias_type='entropy', cond_nch_blck=nch)
        if plot_figs and flag:
            fig = pf.plot_bias(bias=bias_entr, step=step,  per=acr_tr_per)
            fig.savefig(file+'/acr_training_bias_entropy.png', dpi=400,
                        bbox_inches='tight')
            plt.close(fig)
        flag, _, bias_psych, _ =\
            get_bias_single_exp(file, acr_tr_per=acr_tr_per, step=step,
                                bias_type='simple_bias', cond_nch_blck=nch)
        if plot_figs and flag:
            fig = pf.plot_bias(bias=bias_psych, step=step, per=acr_tr_per)
            fig.savefig(file+'/acr_training_bias_psycho.png', dpi=400,
                        bbox_inches='tight')

            plt.close(fig)
    if rebound:
        flag, bs_bef_aft, last_seq = get_rebound_single_exp(file,
                                                            min_num_tr=acr_tr_per,
                                                            cond_nch_blck=nch)
        if plot_figs and flag:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
            plt.plot(last_seq, '+-')
            fig.savefig(file+'/rebound.png', dpi=400, bbox_inches='tight')

    return trans_ac, trans_ae, weights_ac, weights_ae, bias_psych, bias_entr,\
        perf, bs_bef_aft, last_seq


def compute_bias_for_sequences(main_folder, mat_max_n_ch=['2', '8'],
                               period=-10000000, templ_specs=None,
                               template_perf=None, template_rep=None,
                               ax_mns=None, name='rebound',
                               xtickslabels=[''], plot_fig=True):
    if plot_fig:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 4))
        ax.plot([0, 7], [0, 0], '--', color=(.5, .5, .5), lw=0.5)
    all_biases = {}
    for ind_mnch, max_n_ch in enumerate(mat_max_n_ch):
        print('xxxxxxxxxxxxxxxxxxxx')
        print(max_n_ch)
        files =\
            glob.glob(main_folder+'bhvr_data_all*_nch_'+max_n_ch+'.npz')
        biases_mat = []
        for f in files:
            print('-----')
            data = load_behavioral_data(f)
            choice = data['choice'][period:]
            evidence = data['evidence'][period:]
            performance = data['performance'][period:]
            nch_blck_mask = data['nch'][period:] == 2
            tr_hist_mask = None  # data['tr_block'][period:] == 1
            biases = compute_AE_bias_and_rebound(ch=choice, ev=evidence,
                                                 perf=performance,
                                                 nch_mask=nch_blck_mask,
                                                 tr_hist_mask=tr_hist_mask,
                                                 bias_type='simple_bias',
                                                 templ_specs=templ_specs,
                                                 template_perf=template_perf,
                                                 template_rep=template_rep,
                                                 whole_seq=plot_fig)
            biases_mat.append(biases)
            if plot_fig:
                plt.plot(biases, '+-', color=COLORES[ind_mnch], alpha=0.3)
        biases_mat = np.array(biases_mat)
        all_biases[max_n_ch] = biases_mat
        if plot_fig:
            mean_biases = np.mean(biases_mat, axis=0)
            plt.plot(mean_biases, color=COLORES[ind_mnch], lw=2,
                     label='max. num. of ch = '+max_n_ch)
        if ax_mns is not None:
            mean_biases = np.mean(biases_mat, axis=0)
            ax_mns.plot(mean_biases, color=COLORES[ind_mnch], lw=2)

    if plot_fig:
        ax = plt.gca()
        ax.set_xticks(np.arange(8))
        ax.set_xticklabels(xtickslabels)
        ax.set_xlabel('Previous sequence')
        ax.set_ylabel('Repeating bias')
        ax.legend()
        fig.savefig('/home/molano/Dropbox/project_Barna/students/Jorge del Pozo/' +
                    'results/'+name+'.png', dpi=400, bbox_inches='tight')
        plt.close(fig)
    return all_biases


if __name__ == '__main__':
    if len(sys.argv) == 1:
        # main_folder =\
        #     r'C:\Users\jorgi\Dropbox\variable_nch'
        main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
            'diff_probs_nch_blocks/'
    elif len(sys.argv) == 2:
        main_folder = sys.argv[1]
    batch_analysis(main_folder, conds=['ACER', 'A2C', 'ACKTR', 'PPO2'], val='n_ch',
                   step=100000, acr_tr_per=300000, test_retrain='test',
                   analyze_tr_data=True, sv_per_test=100000, n_steps_test=600000)
