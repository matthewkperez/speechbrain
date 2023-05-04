'''
remove 5 HC utts from train and put them in val
Supplement existing data with word_labels
update data w.o neologistic 
'''
import os
import pandas as pd
from collections import Counter

if __name__ == "__main__":
    kaldi_dir = "kd_updated_para"
    aphasia_wrd_labels = f"/z/mkperez/AphasiaBank/{kaldi_dir}/Aphasia/wrd_labels"
    control_wrd_labels = f"/z/mkperez/AphasiaBank/{kaldi_dir}/Control/wrd_labels"
    og_data_dir = "/z/mkperez/speechbrain/AphasiaBank/data/no_paraphasia"
    new_data_dir = f"{og_data_dir}/attn_viz"

    utt2wrd_label = {}
    labels = []
    for fpath in [aphasia_wrd_labels, control_wrd_labels]:
        with open(fpath, 'r') as r:
            for line in r.readlines():
                line = line.strip()
                utt_id = line.split()[0]

                word_label_seq=[]
                for c in line.split()[1:]: 
                    if c == 'x@n':
                        tok = "n"
                    elif c == 'C':
                        tok = c.lower()
                    else:
                        tok = c.split(":")[0]
                    word_label_seq.append(tok)

                # word_label_seq = [c if c == 'C' else c.split(":")[0] for c in line.split()[1:]]
                # word_label_seq 

                # c, p, or n
                word_label_seq = ['c' if w not in ['p', 'n'] else w for w in word_label_seq]
                
                utt2wrd_label[utt_id] = word_label_seq
                labels+=word_label_seq
    print(f"labels: {Counter(labels)}")


    # data dirs for paraphasia
    os.makedirs(new_data_dir,exist_ok=True)

    # make new paraphasia csvs
    remove_HC_ids = ['wright99a-97','MSUC01a-109', 'capilouto33a-494', 'richardson200-69','wright08a-140']
    HD_ids = ['ACWT09a-13','adler15a-546','BU10a-508','kurland01b-249','scale05a-91']
    # for stype in ['train', 'dev','test']:
    for stype in ['dev', 'dev','test']:
        og_file = f"{og_data_dir}/{stype}.csv"
        df = pd.read_csv(og_file)
        og_num_samples = df.shape[0]

        HC_df = df[df['HC_AB'] != 'Control']

        HC_df = HC_df.drop(columns=['aphasia_type','group', 'severity_cat','wav'])
        i = 5100
        print(HC_df.shape)
        print(HC_df[i:i+20])
        exit()


        paraphasia_list = []
        drop_cols = []
        aug_para_wrds= []
        
        # filter paraphasias
        for i,row in df.iterrows():
            if row['ID'] not in utt2wrd_label or len(row['wrd'].split()) != len(utt2wrd_label[row['ID']]):
                drop_cols.append(i)
            else:
                paraphasia_list.append(" ".join(utt2wrd_label[row['ID']]))

                
                aug_para_wrds.append([f"{word}/{para}" for word,para in zip(row['wrd'].split(), utt2wrd_label[row['ID']])])
        


  
        # drop rows
        df = df.drop(drop_cols)
        df['paraphasia'] = paraphasia_list
        df['aug_para'] = aug_para_wrds


        if stype == 'train':
            # remove HC speakers
            df_sub = df[df['ID'].isin(remove_HC_ids)]
            df = df[~df['ID'].isin(remove_HC_ids)]

        elif stype == 'dev':
            # print(df)
            df = pd.concat([df,df_sub])
            # print(df)
            # exit()

        new_num_samples = df.shape[0]

        print(f"{stype}: {og_num_samples} -> {new_num_samples}")
        df.to_csv(f"{new_data_dir}/{stype}.csv")



