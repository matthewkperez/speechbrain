'''
Partition existing data into new csvs (new child directory)
Create new pretrain and finetune csvs
'''
import argparse
import os
import csv
import pandas as pd

def read_csv_fluency(src_csv):
    # labels
    fluency_key={'Control':'Control',
                 'not aphasic': 'Control',
                 'NaN': 'Control',
                 'Global':'Non-Fluent',
                 'TransMotor':'Non-Fluent',
                 'Broca':'Non-Fluent',
                 'Wernicke': 'Fluent',
                 'TransSensory': 'Fluent',
                 'Conduction': 'Fluent',
                 'Anomic': 'Fluent',
                 }
    fluency_dict={}
    df = pd.read_csv(src_csv)
    df['aphasia_type'] = [t.strip() if isinstance(t,str) else 'NaN' for t in df['aphasia_type']]
    for i, row in df.iterrows():

        fluency_val = fluency_key[row['aphasia_type']]

        if fluency_val not in fluency_dict:
            fluency_dict[fluency_val] = []
        fluency_dict[fluency_val].append(row)

    return fluency_dict


def fluency_partition(fluency_dict,tar_dir,fluency_str,stype):

    ft_lst = fluency_dict[fluency_str]
    fluency_dict.pop(fluency_str)
    pt_lst = []
    for _,v in fluency_dict.items():
        pt_lst+=v
    pt_df = pd.concat(pt_lst,axis=1).T
    ft_df = pd.concat(ft_lst,axis=1).T

    # pt
    outfile_pt = f"{tar_dir}/PT_{stype}.csv"
    pt_df.to_csv(outfile_pt)


    # ft
    outfile_ft = f"{tar_dir}/FT_{stype}.csv"
    ft_df.to_csv(outfile_ft)

    


def fluency_split(data_dir):
    # fluent -> finetune for fluent, pt with control/non-fluent
    for stype in ['train', 'dev']:
        src_csv = f"{data_dir}/{stype}.csv"
         # read existing csvs
        fluency_dict = read_csv_fluency(src_csv)
        
        for fluent_str in ['Fluent', 'Non-Fluent']:
            tar_dir = f"{data_dir}/PT_FT-fluency/{fluent_str}"
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            
            fluency_partition(fluency_dict,tar_dir,fluent_str,stype)


            os.symlink(f"{data_dir}/test.csv",f"{tar_dir}/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split_type', help='what you are partitioning on')
    parser.add_argument('data_dir', help='og_dir with csvs')
    args = parser.parse_args()

    if args.split_type=="fluency":
        fluency_split(args.data_dir)
    else:
        print("Error invalid argument")
        exit()

