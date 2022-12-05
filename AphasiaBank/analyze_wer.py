
import os
from tqdm import tqdm
import chaipy.io
import numpy as np
import pandas as pd
import argparse
from hyperpyyaml import load_hyperpyyaml

import seaborn as sns
import matplotlib.pyplot as plt


def detailed_error_sev(filepath,spk2sev,stitle):
    # Output detailed breakdown of WER by severity bins
    overall_wer = -1
    WER_AB = {}
    with open(filepath, 'r') as r:
        for i, line in tqdm(enumerate(r.readlines()), total=len(r.readlines())):
            line = line.strip()
            if i == 0:
                overall_wer = float(line.split()[1])

            elif len(line.split()) == 14 and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                wer = float(line.split()[2])
                spkr_id = utt_id.split("-")[0]

                # enure label is present
                if spkr_id in spk2sev:
                    sev_bin = spk2sev[spkr_id]

                    if sev_bin not in WER_AB:
                        WER_AB[sev_bin] = []
                    WER_AB[sev_bin].append(wer)
    
    # compile sum
    print(f"\n{stitle}")
    sev_list_title = ['mild', 'moderate', 'severe', 'v_severe', 'control', 'unk']
    wer_list = [np.around(np.mean(WER_AB[k]),decimals=2) for k in sev_list_title]
    print(f"{sev_list_title}")
    print(f"{wer_list}")

    # df, index = utt_id, 
    df = pd.DataFrame({
        f'{sev_list_title[0]}': wer_list[0],
        f'{sev_list_title[1]}': wer_list[1],
        f'{sev_list_title[2]}': wer_list[2],
        f'{sev_list_title[3]}': wer_list[3],
        'metric': stitle,
    }, index=[0])
    df = df.set_index('metric')
    return df, overall_wer

def detailed_error_subtype(filepath,spk2subtype,stitle):
    # Output detailed breakdown of WER by severity bins
    WER_AB = {}
    print(spk2subtype)
    # spk2subtype = {k:'NA' if len(v)==2 else k:v for k,v in spk2subtype.items()}
    spk2subtype 
    with open(filepath, 'r') as r:
        for line in tqdm(r.readlines()):
            line = line.strip()
            if len(line.split()) == 14 and line.endswith("]"):
                utt_id = line.split()[0][:-1]
                wer = float(line.split()[2])
                spkr_id = utt_id.split("-")[0]
                sev_bin = spk2subtype[spkr_id]
                if len(sev_bin) == 2:
                    sev_bin = 'not aphasic'
                if sev_bin not in WER_AB:
                    WER_AB[sev_bin] = []
                WER_AB[sev_bin].append(wer)
    
    # compile sum
    print(f"\nSubtype {stitle}")
    wer_list = [np.around(np.mean(WER_AB[k]),decimals=2) for k in sorted(WER_AB.keys())]
    count_list = [len(WER_AB[k]) for k in sorted(WER_AB.keys()) if k not in ['control', 'not aphasic', 'unk']]
    norm_count_list = [np.around(len(WER_AB[k])/sum(count_list),decimals=2) for k in sorted(WER_AB.keys())]
    df = pd.DataFrame({
        'aph_types': sorted(WER_AB.keys()),
        'wer':wer_list,
        'test_pct': norm_count_list,
    })
    print(df)
    # print(f"{sorted(WER_AB.keys())}")
    # print(f"{wer_list}")
    # print(f"{norm_count_list}")

def plot_training_graphs(train_log_file, test_cer, test_wer, result_dir):
    df_lst=[]
    with open(train_log_file, 'r') as r:
        lines = r.readlines()
        for line in lines:
            if line.startswith("epoch: "):
                # read info
                line = line.replace(" -",",")
                sections = line.split(", ")
                epoch = sections[0].split(": ")[-1]
                train_loss = float(sections[3].split(": ")[-1])
                val_loss = float(sections[4].split(": ")[-1])
                val_CER = float(sections[5].split(": ")[-1])
                val_WER = float(sections[6].split(": ")[-1].strip())


                df_loc = pd.DataFrame({
                    'epoch': [epoch],
                    'train_loss': [train_loss],
                    'val_loss': [val_loss],
                    'val_CER': [val_CER],
                    'val_WER': [val_WER]
                })
                df_lst.append(df_loc)
            

    df = pd.concat(df_lst)
    df_melt = pd.melt(df,id_vars=['epoch','val_CER','val_WER'], value_vars=['train_loss','val_loss'])

    # loss
    plt.clf()
    ax = sns.lineplot(data=df_melt, x="epoch", y="value", hue="variable")
    ax.set(title=f"CER: {test_cer} | WER: {test_wer}", xlabel="epoch", ylabel="loss")
    fig = ax.get_figure()
    fig.savefig(f"{result_dir}/loss.png")
    # exit()




def main():
    # output csv to result_dir
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

    parser.add_argument('-r','--result_dir') 
    parser.add_argument('-y','--yaml_file') 
    args = parser.parse_args()

    if args.result_dir:
        result_dir = args.result_dir
    elif args.yaml_file:
        with open(args.yaml_file) as fin:
            hparams = load_hyperpyyaml(fin)
            # print(f"hparams: {hparams}")
            result_dir = hparams['output_folder']
    else:
        print("error: result_dir not found")
        return 0
    # exit()


    WER_file=f"{result_dir}/wer.txt"
    CER_file=f"{result_dir}/cer.txt"
    train_log_file=f"{result_dir}/train_log.txt"
    
    # severity classes
    spk2sev = chaipy.io.dict_read("spk2sevbin")
    spk2subtype = chaipy.io.dict_read("spk2subtype")

    df_wer, overall_wer = detailed_error_sev(filepath=WER_file,spk2sev=spk2sev,stitle="WER")
    df_cer, overall_cer  = detailed_error_sev(filepath=CER_file,spk2sev=spk2sev,stitle="CER")

    print(df_cer)
    print(df_wer)
    df_all = pd.concat([df_cer,df_wer], axis=0)
    df_all.to_csv(f"{result_dir}/error_severity.csv")

    # plot training graphs
    plot_training_graphs(train_log_file, overall_cer, overall_wer, result_dir)


    # detailed_error_subtype(filepath=WER_file,spk2subtype=spk2subtype,stitle="WER")
    # detailed_error_sev(filepath=CER_file,spk2sev=spk2sev,stitle="CER")

    

if __name__ == "__main__":
    main()