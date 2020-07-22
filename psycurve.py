#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 09:46:55 2020

@author: junhuang
"""



import datajoint as dj

dj.config['database.host'] = 'datajoint-public.internationalbrainlab.org'
dj.config['database.user'] = 'nma-ibl-public'
dj.config['database.password'] = 'ibl.pipeline.public.demo'
dj.conn() # explicitly checks if database connection can be established

# behavior data
from nma_ibl import behavior

# analysis result on behavioral data
from nma_ibl import behavior_analyses

# meta information about subjects and sessions
from nma_ibl import subject, acquisition

# A utility function to perform the model fits of the psychometric function
from nma_ibl import psychofit as psy

# some standard packages
import numpy as np
import datetime
import seaborn as sns 
import matplotlib.pyplot as plt

# get psychemtric curves computed for behavioral sessions done by subject with nickname "IBL-T1"
q = behavior_analyses.PsychResultsBlock & (subject.Subject & 'subject_nickname="IBL-T1"')

q & 'session_start_time > "2019-09-15"'

psych_results = q & {'session_start_time': datetime.datetime(2019, 9, 17, 11, 34, 9)}

dict_results = psych_results.fetch(
    'signed_contrasts', 'prob_choose_right', 'n_trials_stim', 'n_trials_stim_right',
    'threshold', 'bias', 'lapse_low', 'lapse_high', as_dict=True)

colors = [[1., 0., 0.], [0., 0., 0.], [0., 0., 1.]]
fig, ax = plt.subplots(1, 1, dpi=200)

for result, color in zip(dict_results, colors):
    pars = [result['bias'], result['threshold'], result['lapse_low'], result['lapse_high']]
    contrasts = result['signed_contrasts'] * 100
    contrasts_fit = np.arange(-100, 100)
    prob_right_fit = psy.erf_psycho_2gammas(pars, contrasts_fit) *100
    sns.lineplot(contrasts_fit, prob_right_fit, color=color, ax=ax)
    sns.lineplot(x=contrasts, y=result['prob_choose_right']*100, err_style="bars", linewidth=0, linestyle='None', mew=0.5,
            marker='.', ci=68, color=color, ax=ax)

# add axis labels
ax.set_xlabel('Signed Contrast (%)')
ax.set_ylabel('Rightward Choice (%)')
