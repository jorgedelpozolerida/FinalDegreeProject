"""
Created on  Feb  2020

@author: jorgedelpozolerida

"""

import helper_functions as hf
import numpy as np
from numpy import logical_and as and_
import matplotlib as mat
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys
sys.path.append(os.path.expanduser("~/neurogym"))
import neurogym.utils.plotting as pl


# GLOBAL VARIABLES
XTICKS = np.array(['1', '2', '3', '4', '5', '6-10'])
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
rebound_tags = ['E' + 'X+'*i for i in range(20)]


# colors
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
COLORS = [mat.cm.rainbow(x) for x in np.linspace(0, 1, 20)]


# SECONDARY FUNCTIONS

def colorbar(mappable, axes=None):
    '''
    Function to create adjusted-to-size color bars in plots.
    '''
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    if axes is not None:
        last_axes = axes
    plt.sca(last_axes)
    return cbar


def reduce_xticks(specs):
    """
    This is just to make xtick shorter for the figures showing the biases/
    performance
    """
    specs = specs.replace('evW_', '')
    return specs


def plot_dashed_lines(minimo, maximo, value=0.5):
    plt.plot([0, 0], [0, 1], '--k', lw=0.2)
    plt.plot([minimo, maximo], [value, value], '--k', lw=0.2)


def define_title(figure, val, num_instances, name=''):
    '''
    Complements and defines suptitle for figure given val and
    num_instances. If name is give info is added to it
    '''
    figure.suptitle(name + ' ( Max N: ' +
                    r'$\bf{' + str(val) + '}$' + ', Instances: '
                    + r'$\bf{' + str(num_instances) + '}$' + ')')


def get_transition_probs_figure(trans_mat, annotate_mat,
                                col_bar='general', title='Ground truth',
                                end_title='', show_values=True,
                                displacement=0):
    '''
    Generates plot for transition matrix.

    Args:
        trans_mat: matrix contianing transition probabilities
        annotate_mat: matrix contianing values to annotate in each
        transition probability box
        choices: unique values of possible choices
        col_bar: str, determines whether to plot colorbar and how. If
        'general', colobar common for all plots is added. If 'specific',
        colorbar for each subplot is added. If None (default), no colorbar.
        blck_tr_hist_id: unique value for trialhistory blocks
        blck_n_ch_id: unique values for nch blocks
        title: 'Ground truth' or 'Choices' for example, to set as title.
        end_title: extra string to add at the end to title, default=''
        displacement: percentage of annotation displacement with respect
        to box size, thought for bif numbers
    '''
    print(trans_mat.shape)
    xy_label_list = [str(i) for i in np.arange(1, trans_mat.shape[2]+1)]
    bin_ = 2 / trans_mat.shape[2]
    yticks = np.linspace(1 - bin_/2, -1 + bin_/2,  trans_mat.shape[2])
    xticks = np.linspace(-1 + bin_/2, 1 - bin_/2, trans_mat.shape[2])
    vmin, vmax = None, None
    if col_bar == 'general':
        # To normalize all plots and general colorbar.
        vmin = 0
        vmax = 1
    f, ax = plt.subplots(ncols=trans_mat.shape[1], nrows=trans_mat.shape[0],
                         figsize=(18, 10))
    plt.subplots_adjust(wspace=0.1, hspace=0.05)
    ax = ax.flatten()
    counter = 0

    for ind_trh in range(trans_mat.shape[0]):
        for ind_nch in range(trans_mat.shape[1]):
            im = ax[counter].imshow(
                trans_mat[ind_trh, ind_nch, :, :],
                cmap='viridis',  # We create a colormap for colorbar.
                extent=[-1, 1, -1, 1],  # To have known reference points.
                vmin=vmin, vmax=vmax)
            ax[counter].set_xlabel(title + ' at trial t+1')
            ax[counter].set_ylabel(title + ' at trial t')
            ax[counter].set_xticks(xticks)
            ax[counter].set_xticklabels(xy_label_list)
            ax[counter].xaxis.tick_top()
            ax[counter].set_yticks(yticks)
            ax[counter].set_yticklabels(xy_label_list)
            ax[counter].yaxis.set_visible(False)
            if ind_trh == 0:
                # set column names in superfigure
                ax[counter].set_title('Effective channels: ' + str(ind_nch),
                                      pad=40, fontsize=12)
                # TODO: set row names in superfigures
            if counter == 0 or counter % trans_mat.shape[1] == 0:
                # share y axis across a row.
                ax[counter].yaxis.set_visible(True)
            if show_values:
                # display value of counts on top
                bin_p = bin_ / 2
                if displacement != 0:
                    disp = ((displacement/100)*bin_p)
                else:
                    disp = 0
                for i in range(trans_mat.shape[2]):
                    for j in range(trans_mat.shape[2]):
                        init = bin_p/2 + i*bin_p - disp
                        end = bin_p/2 + j*bin_p - disp
                        index_r = trans_mat.shape[2] - j - 1
                        annotation = annotate_mat[ind_trh,
                                                  ind_nch, index_r, i]
                        ax[counter].annotate("{:.0f}".format(annotation),
                                             (init, end),
                                             textcoords='axes fraction',
                                             color="w")
            counter += 1
    if col_bar == 'general':
        # common colorbar for figure
        f.colorbar(im, ax=ax.ravel().tolist(), shrink=0.75)
    f.suptitle(title + ' probabilities transition matrix ' + '. N= '
               + str(trans_mat.shape[2]) + end_title,
               fontsize=14)
    return f, ax


def generate_GLM_fig(figsize=(16, 8), constrained_layout=True):
    '''
    Generates and returns empty figure for GML plots
    '''
    f, ax = plt.subplots(nrows=2, ncols=2, sharey=True, figsize=figsize,
                         constrained_layout=constrained_layout)
    for ind_pl, subpl in enumerate(ax.flat):
        if ind_pl == 0 or ind_pl == 2:
            subpl.set(ylabel='GLM weights')
        if ind_pl == 2 or ind_pl == 3:
            subpl.set(xlabel='Steps')
    return f, ax


def generate_bias_fig(figsize=(8, 8), constrained_layout=True):
    """Return empty figure for bias plots."""
    f, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=figsize,
                         constrained_layout=constrained_layout)
    for ind_pl, subpl in enumerate(ax.flat):
        if ind_pl == 0 or ind_pl == 2:
            subpl.set(ylabel='Simple Bias')
        if ind_pl == 2 or ind_pl == 3:
            subpl.set(xlabel='Steps')
    ax[0].set_title('After correct')
    ax[1].set_title('After error')
    return f, ax


def get_ylim(ax, ylim):
    """Calculate and returns ylim desired for ax."""
    ylim = [max(ylim[0], ax.get_ylim()[0]),
            min(ylim[1], ax.get_ylim()[1])]
    return ylim


def plot_zero_line(all_axs):
    """Set x and y limits to axes in all_axs."""
    for ax in all_axs:
        for a in ax:
            xlim = a.get_xlim()
            a.plot(xlim, [0, 0], '--k', lw=0.5)


def customize_rebound_figure(a_seq_mat_cond, ax):
    ax.set_ylabel(' Repeating Bias b')
    xticks = rebound_tags[:a_seq_mat_cond.shape[0]-2]
    xticks.extend([xticks[-1] + 'E', xticks[-1] + 'EC'])
    ax.set_xticks([i for i in range(a_seq_mat_cond.shape[0])])
    ax.set_xticklabels(xticks)


# PRIMARY FUNCTIONS

def plot_transition_probs(folder, sv_fig=True, col_bar='general',
                          show_values=False, displacement=0):
    '''
    Plots ground truth transition probs distribution.
    Args:
        col_bar: str, determines whether to plot colorbar and how. If
        'general', colobar common for all plots is added. If 'specific',
        colorbar for each subplot is added. If None, no colorbar.
        show_values: if True, total counts for each condition shown on

    '''
    if os.path.exists(folder + '/bhvr_data_all.npz'):
        data = np.load(folder + '/bhvr_data_all.npz')
    else:
        data = pl.put_together_files(folder)
    if data:
        ground_truth = data['gt']
        # print('GT: \n', np.unique(ground_truth, return_counts=True))
        choices = np.unique(ground_truth)
        try:
            block_n_ch = data['nch']
        except KeyError:
            block_n_ch = np.full(ground_truth.shape, len(choices))
        block_tr_hist = data['curr_block']
        # get transition blocks
        block_tr_hist = block_tr_hist
        # blck_tr_hist_id = np.unique(block_tr_hist)
        # get number of choices blocks
        block_n_ch = block_n_ch
        # blck_n_ch_id = np.unique(block_n_ch)
        # get choices
        trans_mat, counts_mat = hf.compute_transition_probs_mat(
            ground_truth, choices, block_n_ch,
            block_tr_hist, extra_condition=None)
        f, ax = get_transition_probs_figure(trans_mat, counts_mat,
                                            col_bar=col_bar,
                                            title='Ground truth', end_title='',
                                            show_values=show_values,
                                            displacement=displacement)
        if sv_fig:
            f.savefig(folder + '/transition_matrix.png')
        return trans_mat
    else:
        print('No data in: ', folder)


def plot_performance(folder, window=500, ax=None, perfect_integrator=True,
                     fkwargs={'c': 'tab:blue'}, title='', ytitle='',
                     legend=False, zline=False):
    """
    Plots performance evolution across trials.
    If perfect_integrator=True (default), performance of perfect
    integrator is added in the same plot.
    Folder must
    """
    if os.path.exists(folder + '/bhvr_data_all.npz'):
        data = np.load(folder+'/bhvr_data_all.npz')
    else:
        data = pl.put_together_files(folder)

    if data:
        sv_fig = False
        if ax is None:
            sv_fig = True
            f, ax = plt.subplots(figsize=(8, 8))
        # We extract/calculate necessary data
        gt = data['gt']
        alg_performance = data['performance']
        choices = np.unique(gt)  # Used for title and perfect integrator.
        if isinstance(window, float):
            if window < 1.0:
                window = int(alg_performance.size * window)
        # We average the signal with specified window.
        mean_alg_performance = np.convolve(
            alg_performance, np.ones((window,))/window, mode='valid')
        ax.plot(mean_alg_performance, **fkwargs)  # add color, label etc.

        if perfect_integrator:
            # We calculate and plot performance for perfect integrator.
            cum_stim = data['stimulus']
            cum_stim_abs = np.absolute(cum_stim)
            end_ind = len(choices) + 1  # to select only stimulus data.
            # Choice will be were maximum accumulated value is.
            # (first column correspond to fixation)
            choice_integrator = np.argmax(cum_stim_abs[:, 1:end_ind], axis=1) + 1
            performance_integrator = gt == choice_integrator
            # We plot its performance as a straight line (mean).
            ax.plot([0, len(mean_alg_performance)],
                    np.ones((2,))*np.mean(performance_integrator),
                    **{'color': 'purple'})

        # Other plotting parameters added.
        ax.set_xlabel('trials')
        if not title:
            f.suptitle('Number of choices: ' + str(len(choices)))
        if not ytitle:
            ax.set_ylabel('mean performance (running window' +
                          ' of {:d} trials)'.format(window))
        else:
            ax.set_ylabel(ytitle)
        if legend:
            ax.legend()
        if zline:
            ax.axhline(0, c='k', ls=':')
        if sv_fig:
            f.savefig(folder + '/mean_performance' + '_across_training.png')
    else:
        print('No data in: ', folder)


def plot_choice_transition_probs(folder, after_correct=False,
                                 col_bar='general', sim_agent=False,
                                 prop_wrong=None, period=None,
                                 plot_figs=True, show_values=False,
                                 sv_fig=True, displacement=0):
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
        # print('Number of trials: ', str(data['gt'].shape[0]))
        if period is None:
            period = [0, data['gt'].shape[0]]
        elif isinstance(period, int):
            period = [data['gt'].shape[0]-period, data['gt'].shape[0]]
        # We select necessary data
        ground_truth = data['gt'][period[0]:period[1]]
        choices = np.unique(ground_truth)
        # print('GT: \n', choices, np.unique(ground_truth, return_counts=True))
        if not sim_agent:
            choices_agent = data['choice'][period[0]:period[1]]
            # print('CHOICES: \n', choices_agent,
            #       np.unique(choices_agent, return_counts=True))
            performance = data['performance'][period[0]:period[1]]
        else:
            choices_agent = ground_truth.copy()
            size = (int(choices_agent.shape[0]*prop_wrong),)
            indx = np.random.choice(np.arange(choices_agent.shape[0],),
                                    size=size,
                                    replace=False)
            choices_agent[indx] = np.random.choice(choices, size=size)
            performance = choices_agent == ground_truth
        stim = data['stimulus'][period[0]:period[1]]
        # Manage possible situation when active choices remain constant.
        try:
            block_n_ch = data['nch'][period[0]:period[1]]
        except KeyError:
            block_n_ch = np.full(ground_truth.shape, len(choices))
        block_tr_hist = data['curr_block'][period[0]:period[1]]
        # Get transition blocks
        # blck_tr_hist_id = np.unique(block_tr_hist)
        # Get number of choices blocks
        # blck_n_ch_id = np.unique(block_n_ch)
        # Percentile selection of highest stimuli.
        end_ind_nch = len(choices) + 1
        evidence = np.abs(np.max(stim[:, 1:end_ind_nch], axis=1) -
                          np.mean(stim[:, 1:end_ind_nch], axis=1))
        evidence = np.append(evidence[1:],
                             max(evidence))  # shift back evidence
        percetile_10 = np.percentile(evidence, 10)
        # Select between after error or after correct for bias discrimation.
        titles = ['after error', 'after correct']
        title = titles[after_correct]
        extra_condition = and_(performance == after_correct,
                               evidence <= percetile_10)
        # Matrix to fill for all causistry.
        trans_mat, counts_mat = hf.compute_transition_probs_mat(
            choices_agent, choices, block_n_ch,
            block_tr_hist, extra_condition=extra_condition)
        if plot_figs:
            f, ax = get_transition_probs_figure(trans_mat, counts_mat,
                                                col_bar='general',
                                                title=title + ' choice',
                                                end_title=', period= '
                                                + str(period),
                                                show_values=True,
                                                displacement=displacement)
            if sv_fig:
                f.savefig(folder + '/choice_transition_matrix_' + title + '.png')
        return trans_mat
    else:
        print('No data in: ', folder)
        return None


def plot_glm_weights(weights_ac, weights_ae, tags_mat, step, per,
                     num_tr_back=3, axs=None, linewidth=0.5, nch=None,
                     **kwargs):
    '''
    Plotting function for GLM weights. For each list of tags in tags_mat,
    a figure is created. For eah tag in each list, a subplot is generated.

    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1,
                 'N': 0, 'compared_averages': False,
                 'num_tr_tm': None}
    '''
    plot_opts = {'legend': False, 'lw': .5,  'label': '', 'alpha': 1,
                 'N': 0, 'compared_averages': False,
                 'num_tr_tm': num_tr_back}  # TODO: num_tr_back is passed twice (?)
    plot_opts.update(kwargs)
    weights = [weights_ac, weights_ae]
    regressors = [afterc_cols, aftere_cols]
    l_styl = ['-', '--']
    titles = ['a_c', 'a_e']
    figs = []

    for ind_tag, tags in enumerate(tags_mat):
        if axs is None:
            f, ax = plt.subplots(nrows=len(tags)//2, ncols=2, sharex=True,
                                 sharey=True)
            figs.append(f)
            ax = ax.flatten()
        else:
            ax = axs[ind_tag]
        for ind_cond, cond in enumerate(zip(weights, regressors, l_styl)):
            l_st = cond[2]  # different line style for a_c and a_e
            weights_tmp = np.array(cond[0]).squeeze()  # weights
            rgr_tmp = np.array(cond[1])  # labels for data contained in weights
            for ind_t, tag in enumerate(tags):
                t_contr = and_(tag != 'evidence', tag != 'intercept')
                num_tr_tm = num_tr_back+1 if t_contr else 2
                if plot_opts['num_tr_tm']:
                    num_tr_tm = plot_opts['num_tr_tm']
                for ind_tr in range(1, num_tr_tm + 1):
                    t_tmp = tag+str(ind_tr) if t_contr else tag
                    if t_tmp in rgr_tmp:
                        # Plot tunning
                        # label given (for tunning legend) only when:
                        lbl = (plot_opts['label'] + 'trial lag ' + t_tmp[-1]
                               + ' ' + titles[ind_cond]) if ind_t == 0 and \
                               plot_opts['legend'] else ''
                        color = COLORES[ind_tr-1]
                        # for the case of only averages compared for nchs.
                        if plot_opts['compared_averages']:
                            color = COLORS[int(plot_opts['N'])]
                            lbl = 'N=' + str(plot_opts['N']) \
                                  + ', lag= ' + str(ind_tr) + ' , ' \
                                  + titles[ind_cond]
                            lbl = lbl if ind_t == 0 else ''
                            alpha = 1-(1/(num_tr_tm+1))*(ind_tr-1)
                            plot_opts.update({'alpha': alpha})

                        ax[ind_t].plot(np.arange(weights_tmp.shape[0])*step+per/2,
                                       weights_tmp[:, rgr_tmp == t_tmp],
                                       color=color, linestyle=l_st,
                                       label=lbl, linewidth=plot_opts['lw'],
                                       alpha=plot_opts['alpha'])
                        if lbl != '':
                            ax[ind_t].legend()
                ax[ind_t].set_title(tag)
        # ax[0].legend()
    return figs


def plot_bias(bias, step, per, ax=None, given_cols=False,
              **kwargs):
    """
    plots RNN biases across training conditioned on:
    alt cont + alt (after correct/error)
    rep cont + rep (after correct/error)
    alt cont + rep (after correct/error)
    rep cont + alt (after correct/error)
    The function assumes that a figure has been created
    before it is called.
    """
    if ax is None:
        f, ax = plt.subplots(nrows=1, ncols=2, sharey='row', figsize=(8, 4))
        ax = ax.flatten()
    else:
        f = None

    plot_opts = {'legend': False, 'lw': .5,  'label': '',
                 'plot_io': True, 'plot_fract': False, 'alpha': 1}
    plot_opts.update(kwargs)
    num_trials = step*bias.shape[0]
    # plot bias
    # 2nd dim:
    # 0 --> after error
    # 1 --> after correct
    # 3rd dim:
    # 0 --> alt cont + alt
    # 1 --> rep cont + rep
    # 2 --> alt cont + rep
    # 3 --> rep cont + alt
    if given_cols:
        col1, col2 = given_cols, given_cols
        lw1, lw2 = '--', '-'
    else:
        col1, col2 = rojo, azul
        lw1, lw2 = '-', '-'
    ax[0].plot(np.arange(bias.shape[0])*step+per/2, bias[:, 1, 0],
               color=col1, alpha=plot_opts['alpha'], linestyle=lw1,
               lw=plot_opts['lw'], label='alt '+plot_opts['label'])
    ax[0].plot(np.arange(bias.shape[0])*step+per/2, bias[:, 1, 1],
               color=col2, alpha=plot_opts['alpha'],  linestyle=lw2,
               lw=plot_opts['lw'], label='rep '+plot_opts['label'])
    ax[1].plot(np.arange(bias.shape[0])*step+per/2, bias[:, 0, 0],
               color=col1, alpha=plot_opts['alpha'], lw=plot_opts['lw'],
               linestyle=lw1)
    ax[1].plot(np.arange(bias.shape[0])*step+per/2, bias[:, 0, 1],
               color=col2, alpha=plot_opts['alpha'], lw=plot_opts['lw'],
               linestyle=lw2)
    for a in ax:
        a.set_xlabel('trials')
        a.set_xticks([0, num_trials])
    if plot_opts['legend']:
        ax[1].legend()
    return f


def plot_psycho_curve(x, popt, label='', color=azul, alpha=1, lw=0.5):
    # get the y values for the fitting
    y = hf.probit_lapse_rates(x, popt[0], popt[1],
                              popt[2], popt[3])
    plt.plot(x, y, color=color,  label=label, lw=lw,
             alpha=alpha)
    plt.legend(loc="lower right")
    plot_dashed_lines(-np.max(x), np.max(x))


def plot_kernels(weights_ac, weights_ae, ax=None, alpha=1, lw=1, n_stps_ws=20,
                 inset=None):
    if ax is None:
        _, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9), sharey=True)
        ax = ax.flatten()
    num_xs = len(XTICKS)
    for j, name in enumerate(['T++', 'T-+', 'T+-', 'T--']):
        ax[j].set_xticks(np.arange(num_xs))
        ax[j].set_xlim([-0.1, 5.1])
        xtcks = [name+x for x in XTICKS]
        ax[j].set_xticklabels(xtcks)
        ax[j].axhline(y=0, linestyle=':', c='k')
        ax[j].set_ylabel('weight')
        ax[j].set_title(name)
        indx = np.array([np.where(np.array(afterc_cols) == x)[0]
                         for x in xtcks])
        indx = np.array([x for x in indx if len(x) > 0])
        kernel_ac = np.mean(weights_ac[-n_stps_ws:, 0, indx], axis=0)
        ax[j].plot(np.arange(num_xs-len(indx), num_xs), kernel_ac, color=naranja,
                   label='', linewidth=lw, alpha=alpha)
        indx = np.array([np.where(np.array(aftere_cols) == x)[0]
                         for x in xtcks])
        indx = np.array([x for x in indx if len(x) > 0])
        kernel_ae = np.mean(weights_ae[-n_stps_ws:, 0, indx], axis=0)
        ax[j].plot(np.arange(num_xs-len(indx), num_xs), kernel_ae, color=(0, 0, 0),
                   label='', linewidth=lw, alpha=alpha)
        if name == 'T++':
            reset = 1-np.sum(np.abs(kernel_ae))/np.sum(np.abs(kernel_ac[1:]))
            inset.plot(reset, '+', alpha=alpha, color=(0, 0, 0))
    plt.show()


def plot_GLM_final_kernels(glm_ac, glm_ae, vals, per, step,
                           ylim=[1000, -1000], num_tr_back=3, save_folder=None,
                           compared_GLM_averages=False, **kwargs):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    inset_loc = (0.37, 0.57, 0.075, 0.1)
    all_axs = []  # All axes saved here
    plot_opts = {'legend': False, 'lw': 0.5, 'num_tr_tm': num_tr_back}
    plot_opts.update(kwargs)
    if compared_GLM_averages:
        f, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9), sharey=True)
        ax = ax.flatten()
        ax_inset = plt.axes(inset_loc)
        all_axs.append(ax)
    for val in np.unique(vals):
        glm_ac_cond = glm_ac[vals == str(val)]
        glm_ae_cond = glm_ae[vals == str(val)]
        num_instances = glm_ac_cond.shape[0]
        name1, name2 = '', ''
        if not compared_GLM_averages:
            f, ax = plt.subplots(ncols=2, nrows=2, figsize=(16, 9), sharey=True)
            ax_inset = plt.axes(inset_loc)
            ax = ax.flatten()
            all_axs.append(ax)
            plot_opts.update({'legend': False})
            for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
                if len(glm_ac_tmp) != 0:
                    glm_ae_tmp = glm_ae_cond[ind_glm]
                    name1 = '_all_instances'
                    plot_kernels(glm_ac_tmp, glm_ae_tmp, ax=ax, alpha=1,
                                 inset=ax_inset)
            # Average
            a_glm_ac_cond = hf.get_average_mat_cond(glm_ac_cond)
            a_glm_ae_cond = hf.get_average_mat_cond(glm_ae_cond)
            plot_opts.update({'lw': 2, 'label': 'Average, ',
                              'legend': True})
            plot_kernels(a_glm_ac_cond, a_glm_ae_cond, ax=ax, alpha=1,
                         inset=ax_inset)
            name2 = '_averages'
            plot_opts.update({'lw': 0.5, 'label': 'Average, '})
            plot_opts.update({'legend': True})
            ylim = get_ylim(ax[0], ylim)
            name = 'GLM_kernels_max_N_' + str(val) + name1 + name2 \
                   + '_num_tr_back_' + str(plot_opts['num_tr_tm'])
            define_title(f, val, num_instances, name)
            plot_zero_line(all_axs)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
        else:
            a_glm_ac_cond = hf.get_average_mat_cond(glm_ac_cond)
            a_glm_ae_cond = hf.get_average_mat_cond(glm_ae_cond)
            alpha = float(val)/np.max([float(x) for x in vals])
            plot_kernels(a_glm_ac_cond, a_glm_ae_cond, ax=ax, alpha=alpha, lw=2,
                         inset=ax_inset)
    if compared_GLM_averages:
        ylim = get_ylim(ax[0], ylim)
        f.suptitle('Average GLM for different N ( Max N:'
                   r'$\bf{' + str(val) + '}$' + ')', fontsize=13)
        if save_folder is not None:
            name = 'compared_GLM_kernel_averages_max_N_'  \
                + str(np.unique(vals)[-1]) + '_num_tr_back_' \
                + str(plot_opts['num_tr_tm'])
            plot_zero_line(all_axs)
            f.savefig(save_folder + '/' + name + '.png', dpi=400,
                      bbox_inches='tight')


def plot_GLM_results(glm_ac, glm_ae, vals, tags_mat, per, step,
                     ylim=[1000, -1000], num_tr_back=3, save_folder=None,
                     compared_GLM_averages=False, **kwargs):
    '''
    Given GLM matrices after correct and after error, calculates GLM
    results for val data in vals.

    Args:
        tags_mat: list of lists, each generating one figure. For each
        element in each list a subplot is generated.

    '''
    all_axs = []  # All axes saved here
    plot_opts = {'legend': False, 'lw': 0.5, 'num_tr_tm': num_tr_back}
    plot_opts.update(kwargs)
    if compared_GLM_averages:
        f, ax = generate_GLM_fig()
        all_axs.append(ax.flatten())
        ax = [ax.flatten()]
    for val in np.unique(vals):
        glm_ac_cond = glm_ac[vals == str(val)]
        glm_ae_cond = glm_ae[vals == str(val)]
        num_instances = glm_ac_cond.shape[0]
        name1, name2 = '', ''
        if not compared_GLM_averages:
            f, ax = generate_GLM_fig()
            all_axs.append(ax.flatten())
            ax = [ax.flatten()]
            plot_opts.update({'legend': False})
            for ind_glm, glm_ac_tmp in enumerate(glm_ac_cond):
                if len(glm_ac_tmp) != 0:
                    glm_ae_tmp = glm_ae_cond[ind_glm]
                    name1 = '_all_instances'
                    plot_glm_weights(glm_ac_tmp, glm_ae_tmp, tags_mat,
                                     num_tr_back=num_tr_back,
                                     axs=ax, step=step, per=per,
                                     **plot_opts)
            # Average
            a_glm_ac_cond = hf.get_average_mat_cond(glm_ac_cond)
            a_glm_ae_cond = hf.get_average_mat_cond(glm_ae_cond)
            plot_opts.update({'lw': 2, 'label': 'Average, ',
                              'legend': True})
            plot_glm_weights(a_glm_ac_cond, a_glm_ae_cond, tags_mat,
                             num_tr_back=num_tr_back, axs=ax, step=step,
                             per=per, **plot_opts)
            name2 = '_averages'
            plot_opts.update({'lw': 0.5, 'label': 'Average, '})
            plot_opts.update({'legend': True})
            ylim = get_ylim(ax[0][0], ylim)
            name = 'GLM_results_max_N_' + str(val) + name1 + name2 \
                   + '_num_tr_back_' + str(plot_opts['num_tr_tm'])
            define_title(f, val, num_instances, name)
            plot_zero_line(all_axs)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')
        else:
            a_glm_ac_cond = hf.get_average_mat_cond(glm_ac_cond)
            a_glm_ae_cond = hf.get_average_mat_cond(glm_ae_cond)
            plot_opts.update({'lw': 2, 'N': val,
                              'compared_averages': True,
                              'num_tr_tm': 1})  # Set to 2
            plot_glm_weights(a_glm_ac_cond, a_glm_ae_cond, tags_mat,
                             axs=ax, step=step, per=per, **plot_opts)
    if compared_GLM_averages:
        ylim = get_ylim(ax[0][0], ylim)
        f.suptitle('Average GLM for different N ( Max N:'
                   r'$\bf{' + str(val) + '}$' + ')', fontsize=13)
        if save_folder is not None:
            name = 'compared_GLM_averages_max_N_'  \
                + str(np.unique(vals)[-1]) + '_num_tr_back_' \
                + str(plot_opts['num_tr_tm'])
            plot_zero_line(all_axs)
            f.savefig(save_folder + '/' + name + '.png', dpi=400,
                      bbox_inches='tight')


def plot_bias_results(vals, bias_mat, step, per, ylim=[1000, -1000],
                      save_folder=None, compared_bias=False):
    '''
    Plots bias results of bias_mat for vals.
    '''
    all_axs = []
    if compared_bias:
        f, ax = generate_bias_fig()
        all_axs.append(ax.flatten())
        ax = ax.flatten()
    for val in np.unique(vals):
        bias_mat_cond = bias_mat[vals == str(val)]
        num_instances = len(bias_mat_cond)
        name1, name2 = '', ''
        if not compared_bias:
            f, ax = generate_bias_fig()
            ax = ax.flatten()
            all_axs.append(ax)
            for ind_b, b_mat in enumerate(bias_mat_cond):
                if len(b_mat) != 0:
                    plot_bias(bias=b_mat, step=step, per=per, ax=ax,
                              **{'lw': 0.5})
                    name1 = '_all_instances'

            a_bias_mat_cond = hf.get_average_mat_cond(bias_mat_cond)
            plot_bias(bias=a_bias_mat_cond, step=step, per=per, ax=ax,
                      **{'lw': 2})
            ax[0].legend(['Alt context - Alt', 'Rep context - Rep'])
            name2 = '_averages'
            ylim = get_ylim(ax[0], ylim)
            name = 'bias_results' + name1 + name2 + 'max_N_' + str(val)
            define_title(f, val, num_instances, name)
            plot_zero_line(all_axs)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')

        else:
            given_cols = COLORS[int(val)]
            a_bias_mat_cond = hf.get_average_mat_cond(bias_mat_cond)
            plot_bias(bias=a_bias_mat_cond, step=step, per=per, ax=ax,
                      given_cols=given_cols, **{'lw': 2})

    if compared_bias:
        name = 'compared_average_bias_max_N_' + str(val)
        ylim = get_ylim(ax[0], ylim)
        f.suptitle('Average Simple Bias for different N ( Max N:'
                   r'$\bf{' + str(val) + '}$' + ')', fontsize=13)
        tags = []
        for ind, tag in enumerate(np.repeat(np.unique(vals), 2)):
            if ind % 2 == 0:
                tags.append('N= ' + tag + ', Alt_cont-Alt')
            else:
                tags.append('N= ' + tag + ', Rep_cont-Rep ')
        ax[0].legend(tags)

        plot_zero_line(all_axs)
        if save_folder is not None:
            f.savefig(save_folder + '/' + name + '.png', dpi=400,
                      bbox_inches='tight')


def plot_bias_seqs_results(vals, seq_mat, ylim=[1000, -1000],
                           save_folder=None, compared_seq=False):
    '''
    Plots seq results of seq_mat for vals.
    '''
    all_axs = []
    if compared_seq:
        f, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                             figsize=(14, 7))
        all_axs.append([ax])
    for val in np.unique(vals):
        seq_mat_cond = seq_mat[vals == str(val)]
        num_instances = len(seq_mat_cond)
        name1, name2 = '', ''
        if not compared_seq:
            f, ax = plt.subplots(nrows=1, ncols=1, constrained_layout=True,
                                 figsize=(14, 7))
            all_axs.append([ax])
            for ind_b, s_mat in enumerate(seq_mat_cond):
                if len(s_mat) != 0:
                    ax.plot(s_mat, lw=0.5, color=azul)
                    name1 = '_all_instances'
            a_seq_mat_cond = hf.get_average_mat_cond(seq_mat_cond)
            ax.plot(a_seq_mat_cond, **{'lw': 2})
            customize_rebound_figure(a_seq_mat_cond, ax)
            name2 = '_averages'
            ylim = get_ylim(ax, ylim)
            name = 'seq_results' + name1 + name2 + 'max_N_ ' + str(val)
            define_title(f, val, num_instances, name)
            if save_folder is not None:
                f.savefig(save_folder + '/' + name + '.png', dpi=400,
                          bbox_inches='tight')

        else:
            a_seq_mat_cond = hf.get_average_mat_cond(seq_mat_cond)
            ax.plot(a_seq_mat_cond, **{'lw': 2})
            customize_rebound_figure(a_seq_mat_cond, ax)

    if compared_seq:
        name = 'compared_average_seq'
        ylim = get_ylim(ax, ylim)
        f.suptitle('Average Simple seq for different N, (Max N:'
                   r'$\bf{' + str(val) + '}$' + ')', fontsize=13)
        tags = []
        for ind, tag in enumerate(np.unique(vals)):
            tags.append('N: ' + tag)
        ax.legend(tags)
        plot_zero_line(all_axs)
        if save_folder is not None:
            f.savefig(save_folder + '/' + name + '.png', dpi=400,
                      bbox_inches='tight')


def plot_alg(file, save_folder=None, num_tr_back=3, ylim=[1000, -1000],
             compared_GLM_averages=False, compared_bias=False,
             compared_seq=False):
    '''
    Plots GLM results and Simple Bias for all values in file data.
    Args:
        file: where all data is stored for a given algorithm

    '''
    data = np.load(file, allow_pickle=1)
    per = data['acr_tr_per']
    # per = 100000  # old value when not saved in dict
    step = data['step']
    # step = 50000  # old value when not saved in dict
    vals = data['val_mat']
    # GLM RESULTS
    ################################################################
    # GLM KERNELS
    # Loading data
    glm_ac = data['glm_mats_ac']
    glm_ae = data['glm_mats_ae']
    plot_GLM_final_kernels(glm_ac, glm_ae, vals, per, step,
                           ylim, num_tr_back, save_folder, compared_GLM_averages)
    ###################
    # GLM WEIGHTS ACROSS TRAINING
    # Loading data
    tags_mat = [['T++', 'T-+', 'T+-', 'T--']]
    glm_ac = data['glm_mats_ac']
    glm_ae = data['glm_mats_ae']
    plot_GLM_results(glm_ac, glm_ae, vals, tags_mat, per, step,
                     ylim, num_tr_back, save_folder, compared_GLM_averages)
    ################################################################

    # BIAS RESULTS

    # Loading data
    bias_mat = data['bias_mats_psych']
    plot_bias_results(vals, bias_mat, step, per, ylim, save_folder,
                      compared_bias)

    ################################################################

    # REBOUND RESULTS

    # Loading data
    seq_mat = data['bias_sequence']
    plot_bias_seqs_results(vals, seq_mat, ylim=ylim,
                           save_folder=save_folder,
                           compared_seq=compared_seq)


if __name__ == '__main__':
    #  PLOTTING AND SAVING GLM AND BIAS
    # folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
    #     'diff_probs_nch_blocks/alg_ACER_seed_0_n_ch_14/'
    # hf.process_data(file=folder, acr_tr_per=300000, step=50000)
    # plot_transition_probs(folder, sv_fig=True)
    # asd
    plt.close('all')
    main_folder = '/home/molano/priors/AnnaKarenina_experiments/' +\
        'diff_probs_nch_blocks'
    # main_folder = '/home/molano/priors/AnnaKarenina_experiments/tests/'
    file = main_folder + '/data_ACER.npz'
    # file = '/home/molano/priors/AnnaKarenina_experiments/nalt_trial_hist/' +\
    #     'data_ACER.npz'
    sv_folder = main_folder+'/figures/'
    ylim = [-1000, 1000]
    plot_alg(file, save_folder=sv_folder, num_tr_back=3, ylim=ylim,
              compared_GLM_averages=False, compared_bias=False,
              compared_seq=False)
    plot_alg(file, save_folder=sv_folder, num_tr_back=1, ylim=ylim,
             compared_GLM_averages=True, compared_bias=True, compared_seq=True)
