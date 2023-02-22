'''
python personalized_analysis.py -p /z/mkperez/speechbrain/AphasiaBank/results/personalization/freeze-False \
-b /z/mkperez/speechbrain/AphasiaBank/results/personalization/freeze-False/tr-train_all_val-val_no_kansas


python personalized_analysis.py -b /z/mkperez/speechbrain/AphasiaBank/results/duc_process/No-LM/wav2vec2-large-960h-lv60-self/freeze-False
-t /z/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_fluency/FT-Fluent/No-LM_wav2vec2-large-960h-lv60-self/freeze-False
-f Fluent
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
import jiwer
import Levenshtein
from sklearn.metrics import confusion_matrix


def SD_analysis(args):
    spk2sev = chaipy.io.dict_read("spk2sevbin")
    spk2subtype = chaipy.io.dict_read("spk2subtype")
    spk2dur = compute_speaker_duration()
    subtypes = ['NA' if len(x)==2 else x for _,x in spk2subtype.items()]
    subtype_keys = [x for x,_ in spk2subtype.items()]
    subtype2fluency = {'Broca':False, 'Global':False, 'TransMotor':False, 'NA':True, 
        'Wernicke':True, 'TransSensory':True, 'Anomic':True, 'unk':'unk', 'control':True, 'Conduction':True}
    spk2fluency = {k:subtype2fluency[v] for k,v in zip(subtype_keys, subtypes)}

    # read baseline utts
    baseline_dir = args.baseline
    
    df_list=[]
    baseline_wer = 0
    baseline_tot_wer = 0
    baseline_cer = 0
    baseline_tot_cer = 0
    for error_type in ['cer', 'wer']:
        baseline_file=f"{baseline_dir}/{error_type}.txt"
        ERR_baseline = {}
        Total_baseline = {}
        with open(baseline_file, 'r') as r:
            for i, line in tqdm(enumerate(r.readlines()), total=len(r.readlines())):
                line = line.strip()
                if i == 0:
                    overall_wer = float(line.split()[1])

                elif len(line.split()) == 14 and line.endswith("]"):
                    utt_id = line.split()[0][:-1]
                    wer = float(line.split()[2])
                    spkr_id = utt_id.split("-")[0]
                    

                    if spkr_id not in ERR_baseline:
                        ERR_baseline[spkr_id] = 0
                    if spkr_id not in Total_baseline:
                        Total_baseline[spkr_id] = 0
                    
                    num_errs = int(line.split()[4])
                    total = int(line.split()[6][:-1])
                    ERR_baseline[spkr_id]+=num_errs
                    Total_baseline[spkr_id]+=total

                    if error_type == 'wer':
                        baseline_wer+=num_errs
                        baseline_tot_wer+=total
                    elif error_type == 'cer':
                        baseline_cer+=num_errs
                        baseline_tot_cer+=total

        for spkr, err in ERR_baseline.items():
            total = Total_baseline[spkr]
            
            df_loc = pd.DataFrame({
                'spkr_id': [spkr],
                'error_type': [error_type],
                'error': [round(err/total, 3)],
                'model': ['ref'],
                'duration':[spk2dur[spkr]],
                'fluency': [spk2fluency[spkr]]
            })
            df_list.append(df_loc)

    tot_cer = 0
    tot_wer = 0
    total_length_wer = 0
    total_length_cer = 0
    personalized_files = [f"{args.personalized_root}/{x}" for x in os.listdir(args.personalized_root) if x.startswith("SD-")]
    for p_dir in personalized_files:
        for error_type in ['cer', 'wer']:
            ERR_baseline = {}
            Total_baseline = {}
            p_file=f"{p_dir}/{error_type}.txt"
            with open(p_file, 'r') as r:
                for i, line in tqdm(enumerate(r.readlines()), total=len(r.readlines())):
                    line = line.strip()
                    if i == 0:
                        overall_wer = float(line.split()[1])

                    elif len(line.split()) == 14 and line.endswith("]"):
                        utt_id = line.split()[0][:-1]
                        wer = float(line.split()[2])
                        spkr_id = utt_id.split("-")[0]

                        if spkr_id not in ERR_baseline:
                            ERR_baseline[spkr_id] = 0
                        if spkr_id not in Total_baseline:
                            Total_baseline[spkr_id] = 0
                        
                        num_errs = int(line.split()[4])
                        total = int(line.split()[6][:-1])
                        ERR_baseline[spkr_id]+=num_errs
                        Total_baseline[spkr_id]+=total

                        if error_type == 'cer':
                            total_length_cer+=total
                            tot_cer+=num_errs
                        elif error_type == 'wer':
                            total_length_wer+=total
                            tot_wer+=num_errs


            for spkr, err in ERR_baseline.items():
                total = Total_baseline[spkr]
                
                df_loc = pd.DataFrame({
                    'spkr_id': [spkr],
                    'error_type': [error_type],
                    'error': [round(err/total, 3)],
                    'model': ['target'],
                    'duration':[spk2dur[spkr]],
                    'fluency': [spk2fluency[spkr]]
                })
                df_list.append(df_loc)

    
    df = pd.concat(df_list)
    print(df)

    analysis_dir = f"{args.baseline}/analysis"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    for error_type in ['cer', 'wer']:
        sdf = df[df['error_type'] == error_type]
        sdf.loc[:,'spkr_id'] = [re.findall(r'(\d+)',x)[0] for x in sdf['spkr_id']]
        # exit()
        sdf = sdf.sort_values(['duration'],ascending=True).reset_index(drop=True)
        sdf['duration'] = [round(d,2) for d in sdf['duration']]
        print(sdf['duration'])
        # barplot
        plt.clf()
        ax = sns.barplot(data=sdf, x='duration', y='error', hue='model')
        ax.set(title=f"personalization: {error_type.upper()}", xlabel='duration (min)')
        plt.xticks(rotation=45, fontsize=7)
        plt.tight_layout()
        # ax.set_xlabel("duration",fontsize=30)
        fig = ax.get_figure()
        # save to baseline model
        fig.savefig(f"{analysis_dir}/{error_type}.png")

        sdf.to_csv(f"{analysis_dir}/{error_type}.csv")

    # print totals
    baseline_CER = baseline_cer/baseline_tot_cer
    baseline_WER = baseline_wer/baseline_tot_wer
    print(f"Baseline overall CER: {baseline_CER} | WER: {baseline_WER}")


    kansas_CER = tot_cer/total_length_cer
    kansas_WER = tot_wer/total_length_wer
    print(f"Kansas overall CER: {kansas_CER} | WER: {kansas_WER}")


def make_dicts():
    scores_xlsx_path = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df_scores = pd.read_excel(scores_xlsx_path, sheet_name='Time 1')
    df_scores_r = pd.read_excel(scores_xlsx_path, sheet_name='Repeats')
    df_scores = df_scores[df_scores['Participant ID'].notnull()]
    df_scores_r = df_scores_r[df_scores_r['Participant ID'].notnull()]

    spk2aq_str = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2aq_str2 = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores_r.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}

    spk2type_str = {row['Participant ID']:row['WAB Type'].split("'")[0] for i, row in df_scores.iterrows() if row['WAB Type'] != 'U'}
    spk2type_str2 = {row['Participant ID']:row['WAB Type'].split("'")[0] for i, row in df_scores_r.iterrows() if row['WAB Type'] != 'U'}

    spk2aq_str.update(spk2aq_str2)
    spk2type_str.update(spk2type_str2)

    chaipy.io.dict_write(fname=f"spk2aq",key2val=spk2aq_str)
    chaipy.io.dict_write(fname=f"spk2subtype",key2val=spk2type_str)

def compute_wer(df):
    # Compute overall wer from a given df
    ins_sum = df['ins'].sum()
    del_sum = df['del'].sum()
    sub_sum = df['sub'].sum()
    total_sum = df['total'].sum()


    wer = (ins_sum + del_sum + sub_sum) / total_sum
    return wer

def extract_wer_stats(WER_target_file):
    df_list = []
    WER_target = {} # test_utts: error
    spk2dur = compute_speaker_duration()
    spk2subtype = chaipy.io.dict_read("spk2subtype")
    subtypes = ['NA' if len(x)==2 else x for _,x in spk2subtype.items()]
    subtype_keys = [x for x,_ in spk2subtype.items()]
    subtype2fluency = {'Broca':False, 'Global':False, 'TransMotor':False, 'NA':True, 
        'Wernicke':True, 'TransSensory':True, 'Anomic':True, 'unk':'unk', 'control':True, 'Conduction':True}
    spk2fluency = {k:subtype2fluency[v] for k,v in zip(subtype_keys, subtypes)}

    with open(WER_target_file, 'r') as r:
        lines = r.readlines()
        for i, line in tqdm(enumerate(lines), total=len(lines)):
            line = line.strip()
            if i == 0:
                overall_wer = float(line.split()[1])

            elif len(line.split()) == 14 and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                wer = float(line.split()[2])
                spkr_id = utt_id.split("-")[0]
                ins_count = int(line.split()[7])
                del_count = int(line.split()[9])
                sub_count = int(line.split()[11])
                total_count = int(line.split()[6][:-1])
                # print(line)
                # print(ins_count, del_count, sub_count, total_count)
                # exit()

                WER_target[utt_id] = wer
                df_loc = pd.DataFrame({
                    'spkr_id': [spkr_id],
                    'utt_id': [utt_id],
                    'wer': [wer],
                    'ins':[ins_count],
                    'del': [del_count],
                    'sub': [sub_count],
                    'total':[total_count],
                    'duration':[spk2dur[spkr_id]],
                    'fluency': [spk2fluency.get(spkr_id, "unk")]
                })
                df_list.append(df_loc)
    df = pd.concat(df_list)
    return df

def compute_speaker_duration():
    root = "/z/mkperez/speechbrain/AphasiaBank/data/personalization_torre"
    training_csv = f"{root}/train_all.csv"

    df = pd.read_csv(training_csv)
    # print(df)
    # exit()
    spk2dur={}
    for i, row in df.iterrows():
        if row['spk_id'] not in spk2dur:
            spk2dur[row['spk_id']] = 0
        spk2dur[row['spk_id']]+=row['duration']/60
    # spk2dur = {row['spk_id']:row['duration'] for i, row in df.iterrows()}
    return spk2dur

def fluency_analysis(args):
    filter_var = args.fluency
    baseline_WER_path = f"{args.baseline}/wer.txt"
    target_WER_path = f"{args.target}/wer.txt"
    baseline_CER_path = f"{args.baseline}/cer.txt"
    target_CER_path = f"{args.target}/cer.txt"

    analysis_dir = f"analysis/fluency/{filter_var}"
    if not os.path.exists(analysis_dir):
        os.makedirs(analysis_dir)

    target_wer_df = extract_wer_stats(target_WER_path)
    baseline_wer_df = extract_wer_stats(baseline_WER_path)
    target_cer_df = extract_wer_stats(target_CER_path)
    baseline_cer_df = extract_wer_stats(baseline_CER_path)

    if filter_var == 'Fluent':
        target_wer_df = target_wer_df[target_wer_df['fluency'] == True]
        baseline_wer_df = baseline_wer_df[baseline_wer_df['fluency'] == True]
        target_cer_df = target_cer_df[target_cer_df['fluency'] == True]
        baseline_cer_df = baseline_cer_df[baseline_cer_df['fluency'] == True]

    elif filter_var == 'Non-Fluent':
        target_wer_df = target_wer_df[target_wer_df['fluency'] == False]
        baseline_wer_df = baseline_wer_df[baseline_wer_df['fluency'] == False]
        target_cer_df = target_cer_df[target_cer_df['fluency'] == False]
        baseline_cer_df = baseline_cer_df[baseline_cer_df['fluency'] == False]

    tgt_wer = compute_wer(target_wer_df)
    base_wer = compute_wer(baseline_wer_df)
    tgt_cer = compute_wer(target_cer_df)
    base_cer = compute_wer(baseline_cer_df)

    
    stats_file = f"{analysis_dir}/fluency_stats.txt"
    with open(stats_file, 'w') as w:
        w.write(f"Target: {target_WER_path}\n")
        w.write(f"Baseline: {baseline_WER_path}\n")
        w.write(f"Target WER: {tgt_wer} | Base WER: {base_wer}\n")
        w.write(f"Target CER: {tgt_cer} | Base CER: {base_cer}\n")
    

'''
Transcript analysis
'''
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

def transcript_wer_analysis(wdir, wer_file,n):
    '''
    used in general case to examine word errors between transcripts
    output n most common missed words
    most common missed words

    most common missed words for each speaker
    '''
    if not os.path.exists(wdir):
        os.makedirs(wdir)

    df, char_df = read_transcript_wer(wer_file)

    # distribution of I, D, S in df
    dys_series = df['dys'].value_counts()
    dys_series.to_csv(f"{wdir}/word_IDS_distribution.csv")


    # Confusion matrix of character sub
    char_df = char_df[~char_df['pred_c'].isin([' ',';','<','>'])]
    gt = char_df['gt_c'].values
    pred = char_df['pred_c'].values
    print(f"gt: {sorted(set(gt))}")
    print(f"pred: {sorted(set(pred))}")
    label_union = sorted(list(set(gt) | set(pred)))
    cm = confusion_matrix(gt,pred,labels=label_union)
    cm_pct = cm / np.sum(cm)
    # print(cm_pct)
    
    cross_tab = pd.crosstab(gt, pred, normalize='index')
    # convert to 2d matrix with counts
    plt.clf()
    # ax = sns.heatmap(cm_pct, annot=False, vmax=0.01, vmin=0)
    ax = sns.heatmap(cross_tab, annot=False)
    ax.set(xlabel="Pred", ylabel="Ground Truth", title="Character Substitution Errors")
    fig = ax.get_figure()
    fig.savefig(f"{wdir}/char_heatmap.png")



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
    err_df = err_df[err_df['total_count'] > 100] # filter rare words
    err_df=err_df.drop(columns='total_count')

    # filter top and bottom
    top_n = err_df.head(n)
    bot_n = err_df.tail(n)
    result_df = pd.concat([top_n,bot_n])
    result_df.to_csv(f"{wdir}/top_word_errs.csv")
     
def pk_dys_df(gt,target):
    # return df
    with open(gt, 'r') as r:
        gt_lines = sorted([x.strip() for x in r.readlines()])
    # print(gt_lines)
    with open(target, 'r') as r:
        tar_lines = sorted([x.strip() for x in r.readlines()])
    
    # print(gt_lines[:5], tar_lines[:5])
    print(len(gt_lines), len(tar_lines))
    # assert len(gt_lines) == len(tar_lines)
    g=0
    t=0
    df_lst=[]
    cer_edits=0
    cer_total=0
    for g,_ in enumerate(tqdm(gt_lines)):
        g_spkr, gline = gt_lines[g].split(" ", 1)

        if len(tar_lines[t].split()) > 1:
            t_spkr, tline = tar_lines[t].split(" ", 1)
        else:
            t_spkr = tar_lines[t]
            tline = ""

        if g_spkr == t_spkr:
            # print(gline)
            # print(tline)

            edits = Levenshtein.distance(gline, tline)
            # cer = edits/len(gline)
            cer_edits+=edits
            cer_total+=len(gline)

            # measures = jiwer.compute_measures(gline, tline)
            # print(measures)
            # exit()

            gt_processed, tar_processed = jiwer.measures._preprocess([gline],[tline],jiwer.wer_default,jiwer.wer_default)
            # print(gt_processed, tar_processed)
            editops = Levenshtein.editops(gt_processed[0], tar_processed[0])

            # print(editops)
            ins_del_sub_dict = {'insert': 'I', 'replace': 'S', 'delete': 'D'}
            editops_dict = {e[1]:ins_del_sub_dict[e[0]] for e in editops if e[0] != 'insert'} # dont include insertions
            # exit()
            for i,x in enumerate(gline.split()):
                if i in editops_dict:
                    df_loc = pd.DataFrame({
                        'spkr': [g_spkr],
                        'word': [x],
                        'dys': [editops_dict[i]]
                    })
                else:
                    df_loc = pd.DataFrame({
                        'spkr': [g_spkr],
                        'word': [x],
                        'dys': ['=']
                    })
                df_lst.append(df_loc)
            g+=1
            t+=1
        else:
            g+=1
    
    df = pd.concat(df_lst)
    return df, cer_edits/cer_total

def pk_transcript_wer_analysis(wdir, gt, target, n,w):
    '''
    for pytorch_kaldi docs
    used in general case to examine word errors between transcripts
    output n most common missed words
    most common missed words

    most common missed words for each speaker
    '''
    if not os.path.exists(wdir):
        os.makedirs(wdir)


    
    # df = ins, del, corr
    df,cer = pk_dys_df(gt,target)


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
    err_df = err_df[err_df['total_count'] > 100] # filter rare words
    err_df=err_df.drop(columns='total_count')

    # filter top and bottom
    top_n = err_df.head(n)
    bot_n = err_df.tail(n)
    result_df = pd.concat([top_n,bot_n])
    result_df.to_csv(f"{wdir}/top_word_errs.csv")

    print(f"cer: {cer}")
    w.write(f"cer: {cer}\n")

def main():
    # output csv to result_dir
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

    parser.add_argument('-b','--baseline') # paired
    parser.add_argument('-p','--personalized_root') # paired
    parser.add_argument('-t','--target')  # paired
    parser.add_argument('-f','--fluency')  # paired
    parser.add_argument('-pk','--pytorch_kaldi')  # paired
    args = parser.parse_args()

    make_dicts()

    if args.baseline and args.target and args.fluency:
        '''
        python personalized_analysis.py -b /z/mkperez/speechbrain/AphasiaBank/results/duc_process/No-LM/wav2vec2-large-960h-lv60-self/freeze-False -t /z/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_fluency/FT-Non-Fluent/No-LM_wav2vec2/freeze-False -f Non-Fluent
        '''
        print("fluency analysis")
        fluency_analysis(args)
    elif args.baseline and args.target:
        print("paired analysis")
        paired_analysis(args)
    elif args.baseline and args.personalized_root:
        print("personalization")
        SD_analysis(args)
    elif args.target:
        '''
        python personalized_analysis.py -t /y/mkperez/speechbrain/AphasiaBank/results/duc_process_ES/No-LM/hubert-large-ls960-ft/freeze-True
        '''
        model = args.target.split("/")[8]
        analysis_dir = f"analysis/transcript/{model}"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        with open(f"{analysis_dir}/README", 'w') as w:
            w.write(f"{args.target}\n")

        transcript_wer_analysis(analysis_dir, f"{args.target}/wer.txt",5)
    elif args.pytorch_kaldi:
        '''
        python personalized_analysis.py -pk /z/mkperez/pytorch-kaldi/exp/MTL_AB_BLSTM_mfb_filter-10s/decode_AB_test_out_senone/scoring_kaldi/best_wer
        '''
        model = args.pytorch_kaldi.split("/")[5]
        analysis_dir = f"analysis/transcript/{model}"
        if not os.path.exists(analysis_dir):
            os.makedirs(analysis_dir)
        with open(f"{analysis_dir}/README", 'w') as w:
            w.write(f"{args.pytorch_kaldi}\n")
        
            # get gt file
            score_kaldi_dir = "/".join(args.pytorch_kaldi.split("/")[:-1])
            gt_file = f"{score_kaldi_dir}/test_filt.txt"
    
            # get best target file
            with open(args.pytorch_kaldi, 'r') as r:
                line= r.readline()
                best_wer_file = line.split()[-1]
            penality_weight = best_wer_file.split("_")[-1]
            wer_num = best_wer_file.split("_")[-2]
            target_file=f"{score_kaldi_dir}/penalty_{penality_weight}/{wer_num}.txt"
            pk_transcript_wer_analysis(analysis_dir, gt_file, target_file,5,w)

        

    else:
        print("error: result_dir not found")
        return 0





    

if __name__ == "__main__":
    main()