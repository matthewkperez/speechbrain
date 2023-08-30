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
            
            partition_csv(fluency_dict,tar_dir,fluent_str,stype)


            os.symlink(f"{data_dir}/test.csv",f"{tar_dir}/test.csv")


def partition_csv(inp_dict,tar_dir,inp_str,stype):

    ft_lst = inp_dict[inp_str]
    inp_dict.pop(inp_str)
    pt_lst = []
    for _,v in inp_dict.items():
        pt_lst+=v
    pt_df = pd.concat(pt_lst,axis=1).T
    ft_df = pd.concat(ft_lst,axis=1).T

    # pt
    outfile_pt = f"{tar_dir}/PT_{stype}.csv"
    pt_df.to_csv(outfile_pt)


    # ft
    outfile_ft = f"{tar_dir}/FT_{stype}.csv"
    ft_df.to_csv(outfile_ft)

def read_csv_subtype(src_csv):
    # create dict = {subtype:}
    subtype_dict={}
    df = pd.read_csv(src_csv)
    df['aphasia_type'] = [t.strip() if isinstance(t,str) else 'NaN' for t in df['aphasia_type']]
    for i, row in df.iterrows():
        # print(row['wav'])
        # exit()
        if row['aphasia_type'] not in subtype_dict:
            subtype_dict[row['aphasia_type']] = []
        subtype_dict[row['aphasia_type']].append(row)

    return subtype_dict

def subtype_split(data_dir):
    for stype in ['train', 'dev']:
        src_csv = f"{data_dir}/{stype}.csv"
         # read existing csvs
        subtype_dict = read_csv_subtype(src_csv)

        subtypes_FT = [s for s in subtype_dict.keys() if s not in ['not aphasic', 'NaN', 'Control']]
        for subtype_str in subtypes_FT:
            tar_dir = f"{data_dir}/PT_FT-subtype/{subtype_str}"
            if not os.path.exists(tar_dir):
                os.makedirs(tar_dir)
            
            partition_csv(subtype_dict,tar_dir,subtype_str,stype)

            if not os.path.exists(f"{tar_dir}/test.csv"):
                os.symlink(f"{data_dir}/test.csv",f"{tar_dir}/test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('split_type', help='what you are partitioning on')
    parser.add_argument('data_dir', help='og_dir with csvs')
    args = parser.parse_args()

    if args.split_type=="fluency":
        fluency_split(args.data_dir)
    elif args.split_type=="subtype":
        subtype_split(args.data_dir)
    else:
        print("Error invalid argument")
        exit()

