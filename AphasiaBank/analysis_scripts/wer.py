'''
Given list of wer.txt outputs from SB
Generate seaborn plots of WER based on severity
Plot performance as a function of utt-length
'''

import os
from tqdm import tqdm
import chaipy.io
import numpy as np
import pandas as pd
import argparse
from hyperpyyaml import load_hyperpyyaml

import seaborn as sns
import matplotlib.pyplot as plt
import re


def generate_wer_summary(filepath,spk2subtype,spk2sev,spk2aq):
    '''
    Output dataframe with subtype, severity, and wer for each utt
    filepath = speechbrain wer.txt filepath
    spk2subtype = dict
    spk2sev = dict
    stitle = str (CER or WER)
    '''

    df_lst = []
    with open(filepath, 'r') as r:
        for line in tqdm(r.readlines()):
            line = line.strip()
            if len(line.split()) == 14 and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                wer = float(line.split()[2])
                spkr_id = utt_id.split("-")[0]


                # need both severity and subtype to add
                if spkr_id in spk2subtype and spkr_id in spk2sev:
                    subtype = spk2subtype[spkr_id]
                    if len(subtype) == 2:
                        subtype = 'not aphasic'

                    
                    aq = float(spk2aq[spkr_id])
                    sev = spk2sev[spkr_id]




                    df_loc = pd.DataFrame({
                        'subtype': [subtype],
                        'spkr': [spkr_id],
                        'wer': [wer],
                        'aq': [aq],
                        'sev': [sev],
                    })
                    df_lst.append(df_loc)

    df = pd.concat(df_lst)
    return df

def speaker_df_average(inp_df):
    '''
    Take utt-level df
    return wer averaged over speakers
    '''
    df_list = []
    for spkr in inp_df.spkr.unique():
        spkr_df = inp_df[inp_df['spkr'] == spkr]
        mean_wer = spkr_df.wer.mean()
        
        df_loc = pd.DataFrame({
            'spkr': [spkr],
            'spkr_wer': [mean_wer],
            'subtype': [spkr_df.iloc[0].subtype],
            'aq': [round(spkr_df.iloc[0].aq,2)],
            'sev': [spkr_df.iloc[0].sev],

        })
        df_list.append(df_loc)

        # if spkr_df.iloc[0].aq > 95:
        #     df_loc = pd.DataFrame({
        #         'spkr': [spkr],
        #         'spkr_wer': [mean_wer],
        #         'subtype': [spkr_df.iloc[0].subtype],
        #         'aq': [round(spkr_df.iloc[0].aq,2)],
        #         'sev': ['>95'],
        #     })
        #     df_list.append(df_loc)

    df = pd.concat(df_list)
    return df

def severity_breakdown(df):
    '''
    Input: df is averaged per speaker
    Output: 
        1) barplot to visualize performance differences across methods
        2) dataframe table with statistical significance between methods
    '''

    # plot severity (1)
    plt.clf()
    ax = sns.barplot(data=df, x='sev', y='spkr_wer', hue='exp')
    ax.set(xlabel="Severity (AQ)", ylabel="WER", title="WER Comparison")
    fig = ax.get_figure()
    fig.savefig(f"{analysis_dir}/barplot_sevcat.png")

    # plot Subtype (1)
    plt.clf()
    ax = sns.barplot(data=df, x='sev', y='subtype', hue='exp')
    ax.set(xlabel="Subtype", ylabel="WER", title="WER Comparison")
    fig = ax.get_figure()
    fig.savefig(f"{analysis_dir}/barplot_subtype.png")





if __name__ == "__main__":
    # output dir
    analysis_dir = "/z/mkperez/speechbrain/AphasiaBank/analysis/WER_Severity"
    os.makedirs(analysis_dir, exist_ok=True)
    
    # load const
    spk2aq = chaipy.io.dict_read("../spk2aq")
    spk2sev = chaipy.io.dict_read("../spk2sevbin")
    spk2subtype = chaipy.io.dict_read("../spk2subtype")

    # wer_files = [f"wer_dir/{f}" for f in os.listdir("wer_dir")]
    wer_files = os.listdir("wer_dir")
    
    exp_list = []
    for f in wer_files:
        exp_title = f.split(".")[0]
        wer_path = f"wer_dir/{f}"

        df_exp = generate_wer_summary(wer_path,spk2subtype,spk2sev,spk2aq)

        spkr_df_avg = speaker_df_average(df_exp)
        # exit()

        spkr_df_avg['exp'] = exp_title
        exp_list.append(spkr_df_avg)
    
    spkr_df = pd.concat(exp_list)
    severity_breakdown(spkr_df)
