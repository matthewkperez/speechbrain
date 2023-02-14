'''
Take csv and create .txt files for RNNLM script
'''

import argparse
import yaml
import os
import shutil
from hyperpyyaml import load_hyperpyyaml,dump_hyperpyyaml
import pandas as pd
import json



def create_LM_text(new_LM_text, df):
    with open(new_LM_text, 'w') as w:
        df = pd.read_csv(input_csv)
        # print(df['wrd'])
        # exit()
        for wrd in df['wrd']:
            w.write(f"{wrd}\n")

def create_tokenizer_json(new_LM_json,df):
    # print(df)
    # print(df.columns)
    # exit()
    d = {}
    # {"kurland28a-205":{"words":"oh god\n","dataset":"Aphasia","spk_id":"kurland28a","wav":"\/y\/mkperez\/AphasiaBank\/data\/seg_audio_HF\/data\/kurland28a-205.wav","length":1.979}
    for i, row in df.iterrows():
        d[row['ID']] = {'words':row['wrd'], 'wav':row['wav']}
        

    json_obj = json.dumps(d)
    with open(new_LM_json, 'w') as w:
        w.write(json_obj)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--results_dir')
    parser.add_argument('-d', '--data_dir')
    args = parser.parse_args()


    for subtype in ['train', 'dev', 'test']:
        new_LM_text = f"{args.data_dir}/LM_{subtype}.txt"
        new_LM_json = f"{args.data_dir}/tokenizer_{subtype}.json"
        input_csv = f"{args.data_dir}/{subtype}.csv"
        df = pd.read_csv(input_csv)

        create_LM_text(new_LM_text, df)

        create_tokenizer_json(new_LM_json,df)