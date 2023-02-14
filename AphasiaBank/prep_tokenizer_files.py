

import argparse
import yaml
import os
import shutil
from hyperpyyaml import load_hyperpyyaml,dump_hyperpyyaml
import pandas as pd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # parser.add_argument('-r', '--results_dir')
    parser.add_argument('-d', '--data_dir')
    args = parser.parse_args()


    for subtype in ['train', 'dev', 'test']:
        new_LM_text = f"{args.data_dir}/LM_{subtype}.txt"
        input_csv = f"{args.data_dir}/{subtype}.csv"
        print(new_LM_text)
        with open(new_LM_text, 'w') as w:
            df = pd.read_csv(input_csv)
            # print(df['wrd'])
            # exit()
            for wrd in df['wrd']:
                w.write(f"{wrd}\n")
    