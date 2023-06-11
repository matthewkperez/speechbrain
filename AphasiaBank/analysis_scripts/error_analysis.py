'''
Output Top n errors

Output top n errors for each subtype
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
from sklearn.metrics import confusion_matrix
import jiwer
import Levenshtein

spk2subtype = chaipy.io.dict_read('spk2subtype')
spk2subtype = {k:('NA' if len(v) == 2 else v) for k,v in spk2subtype.items()}

def read_transcript_wer(wer_file):
    '''
    return df with dysfluency count
    '''
    df_list = []
    char_df_list = []
    with open(wer_file, 'r') as r:
        stage = 0
        dys_dict = {'ins':{}, 'del':{}, 'sub':{}, 'err':{}}
        lines = r.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            line = line.strip()
            if len(line.split()) == 14 and line.endswith("]"):
                spkr_id = line.split(",")[0]
                stage = 1
            elif stage==1:
                # ground truth
                ground_truth_arr = [x.strip() for x in line.split(";")]
                stage=2
            elif stage == 2:
                # dys
                dys_arr = [x.strip() for x in line.split(";")]
                assert(len(dys_arr) == len(ground_truth_arr))
                stage = 3

            elif stage==3:
                # prediction
                prediction_arr = [x.strip() for x in line.split(";")]
                stage=0
                # print(ground_truth_arr)
                # print(dys_arr)
                # print(prediction_arr)
                for gt, dys, pred in zip(ground_truth_arr, dys_arr, prediction_arr):
                    gt = gt.strip()
                    dys = dys.strip()
                    # dys_key = {'I':'ins', 'D':'del', 'S':'sub'}

                    if dys == 'S':
                        edits = Levenshtein.editops(pred, gt)
                        for tup in edits:
                            edit_code = tup[0]
                            if edit_code == 'replace':
                                char_df_loc = pd.DataFrame({
                                    'spkr': [spkr_id],
                                    'dys': [dys],
                                    'word': [gt],
                                    'pred_c': [pred[tup[1]]],
                                    'gt_c': [gt[tup[2]]],
                                })
                                char_df_list.append(char_df_loc)
            
                    elif dys == '=':
                        # print(gt, pred)
                        for i in range(len(gt)):
                            char_df_loc = pd.DataFrame({
                                'spkr': [spkr_id],
                                'dys': [dys],
                                'word': [gt],
                                'pred_c': [pred[i]],
                                'gt_c': [gt[i]],
                            })
                            char_df_list.append(char_df_loc)




                    df_loc = pd.DataFrame({
                            'spkr': [spkr_id],
                            'dys': [dys],
                            'word': [gt],
                    })
                    df_list.append(df_loc)



    df = pd.concat(df_list)
    char_df = pd.concat(char_df_list)
    return df,char_df

def word_errors(df, min_count=10):
    # compute top/bot n words
    word_group = df.groupby("word")
    df_lst = []
    for k in tqdm(word_group.groups.keys()):
        word_df = word_group.get_group(k)
        corr = len(word_df[word_df['dys'] == '='])
        err = len(word_df[word_df['dys'] != '='])

        df_loc = pd.DataFrame({
            'word': [k],
            'err_count': [err],
            # 'correct_count': [corr],
            'total_count': [corr+err],
            'err_pct': [err/(corr+err)]
        })
        df_lst.append(df_loc)
    err_df = pd.concat(df_lst).sort_values(by=['err_pct'])
    err_df = err_df[err_df['total_count'] > min_count] # filter rare words
    err_df=err_df.drop(columns='total_count')
    err_df = err_df[err_df['word'] != '<eps>']

    return err_df

def transcript_wer_analysis(model_path,n):
    '''
    used in general case to examine word errors between transcripts
    output n most common missed words
    most common missed words

    most common missed words for each speaker
    '''

    wer_file = f"{model_path}/wer.txt"

    wdir = f"{model_path}/analysis"
    os.makedirs(wdir, exist_ok=True)


    # df, char_df = read_transcript_wer(wer_file)
    # df.to_csv("analysis_scripts/df_load.csv")
    df = pd.read_csv("analysis_scripts/df_load.csv")


    # distribution of I, D, S in df
    dys_series = df['dys'].value_counts()
    dys_series.to_csv(f"{wdir}/word_IDS_distribution.csv")


    # # Confusion matrix of character sub
    # char_df = char_df[~char_df['pred_c'].isin([' ',';','<','>'])]
    # gt = char_df['gt_c'].values
    # pred = char_df['pred_c'].values
    # # print(f"gt: {sorted(set(gt))}")
    # # print(f"pred: {sorted(set(pred))}")
    # label_union = sorted(list(set(gt) | set(pred)))
    # cm = confusion_matrix(gt,pred,labels=label_union)
    # cm_pct = cm / np.sum(cm)
    # # print(cm_pct)
    
    # cross_tab = pd.crosstab(gt, pred, normalize='index')
    # # convert to 2d matrix with counts
    # plt.clf()
    # # ax = sns.heatmap(cm_pct, annot=False, vmax=0.01, vmin=0)
    # ax = sns.heatmap(cross_tab, annot=False)
    # ax.set(xlabel="Pred", ylabel="Ground Truth", title="Character Substitution Errors")
    # fig = ax.get_figure()
    # fig.savefig(f"{wdir}/char_heatmap.png")


    # # compute top/bot n words
    # err_df = word_errors(df)
    # # filter top and bottom
    # top_n = err_df.head(n)
    # bot_n = err_df.tail(n)
    # result_df = pd.concat([top_n,bot_n])
    # result_df.to_csv(f"{wdir}/top_word_errs.csv")


    ### subtype
    df = df.reset_index()
    drop_rows = []
    # print(spk2subtype)
    # exit()
    for i, row in df.iterrows():
        spkr = row['spkr'].split("-")[0]
        # print(spkr)
        if spkr not in spk2subtype:
            drop_rows.append(i)
    df = df.drop(drop_rows)
    

        
    df['subtype'] = [spk2subtype[s.split('-')[0]] for s in df['spkr']]
    print(f"df: {df}")


    subtype_group = df.groupby("subtype")
    df_lst = []
    viewer_friendly = []
    for k in tqdm(subtype_group.groups.keys()):
        subtype_df = subtype_group.get_group(k)
        err_df = word_errors(subtype_df,10)
        # # filter top and bottom
        top_n = err_df.head(n) # best
        bot_n = err_df.tail(n) # worst

        # detailed view
        bot_n['subtype'] = k
        df_lst.append(bot_n)

        # Viewer Friendly (words only)
        if 'word' in bot_n:
            worst_words = list(bot_n['word'])
            # print(worst_words)
            # exit()
            vf_df = pd.DataFrame({
                'subtype': [k],
                'w0': [worst_words[0]],
                'w1': [worst_words[1]],
                'w2': [worst_words[2]],
                'w3': [worst_words[3]],
                'w4': [worst_words[4]],
            })
            viewer_friendly.append(vf_df)
    subtype_results = pd.concat(df_lst)
    vf_subtype_results = pd.concat(viewer_friendly)

    

    #transform
    subtype_results = subtype_results.drop(columns=['err_count','err_pct'])
    print(subtype_results)
    print(vf_subtype_results)
    
    

     

if __name__ == "__main__":
    # model_path = "/z/mkperez/speechbrain/AphasiaBank/results/no_paraphasia/S2S/wavlm-large-GRU/CTC-LE-newBob-1e-4"
    # model_path = "/z/mkperez/speechbrain/AphasiaBank/results/no_paraphasia/S2S/wavlm-large-Transformer/CTC-LE-newBob-1e-4"
    # model_path = "/z/mkperez/speechbrain/AphasiaBank/results/no_paraphasia/Enc-only/wavLM-large"
    model_path = "/z/mkperez/speechbrain/AphasiaBank/results/no_paraphasia/S2S/wavlm-large-Transformer/CTC-LE-DoubleChar"
    # model = model_path.split("/")[-2]
    transcript_wer_analysis(model_path,5)
