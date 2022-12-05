'''
Yaml helper
'''

import argparse
import yaml
import os
import shutil
from hyperpyyaml import load_hyperpyyaml,dump_hyperpyyaml
from io import StringIO 
import torch


def create_yaml_base(args):
    # Create new yaml and return path to generated yaml
    # base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base_torre.yaml"
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(exp_name, sev_arg)
    
    # outfile = f"/y/mkperez/speechbrain/AphasiaBank/hparams/{exp_name}/{sev_arg}.yaml"
    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    shutil.copyfile(base_yaml, outfile)

    sev_num = 4
    min_num = -1
    if sev_arg.startswith("tier"):
        sev_title = sev_arg.split("-")[-1]
        sev_dict = {'control':0, 'mild':1, 'moderate':2, 'severe':3, 'v_severe':4, 'unk':-1}
        sev_num = sev_dict[sev_title]
        min_num = 0

    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('base_PLACEHOLDER', f"{exp_name}/{sev_arg}")
        filedata = filedata.replace('max_sev_PLACEHOLDER', f"{sev_num}")
        filedata = filedata.replace('min_sev_PLACEHOLDER', f"{min_num}")

        with open(outfile,'w') as fout:
            fout.write(filedata)


def create_yaml_torre(args):
    # Create new yaml and return path to generated yaml
    base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base_torre.yaml"
    # base_yaml = "/y/mkperez/speechbrain/AphasiaBank/hparams/base.yaml"
    exp_name = args.exp_name
    sev_arg = args.severity
    print(exp_name, sev_arg)
    
    # outfile = f"/y/mkperez/speechbrain/AphasiaBank/hparams/{exp_name}/{sev_arg}.yaml"
    outfile = args.outfile
    outdir = "/".join(outfile.split("/")[:-1])
    os.makedirs(f"{outdir}", exist_ok=True)
    
    shutil.copyfile(base_yaml, outfile)


    
    if args.severity == "freeze_w2v":
        freeze = "True"
        grad_acc = "4"
        batch_size = "32"
        epochs = "20"
        lr_w2v2 = "0.0001"
        lr_decoder = "0.9"
    elif args.severity == "last2_w2v":
        freeze = "False"
        grad_acc = "16"
        batch_size = "4"
        epochs = "30"
        lr_w2v2 = "0.001"
        lr_decoder = "0.9"
        mtl = "True"



    # replace with raw text
    with open(outfile) as fin:
        filedata = fin.read()
        filedata = filedata.replace('freeze_PLACEHOLDER', f"{freeze}")
        filedata = filedata.replace('grad_acc_PLACEHOLDER', f"{grad_acc}")
        filedata = filedata.replace('batch_size_PLACEHOLDER', f"{batch_size}")
        filedata = filedata.replace('epoch_PLACEHOLDER', f"{epochs}")
        filedata = filedata.replace('lr_w2v2_PLACEHOLDER', f"{lr_w2v2}")
        filedata = filedata.replace('lr_decoder_PLACEHOLDER', f"{lr_decoder}")
        filedata = filedata.replace('mtl_PLACEHOLDER', f"{mtl}")

        with open(outfile,'w') as fout:
            fout.write(filedata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

    parser.add_argument('-e','--exp_name') 
    parser.add_argument('-s', '--severity')
    parser.add_argument('-o', '--outfile')

    args = parser.parse_args()

    if args.exp_name == "severity_training":
        create_yaml_base(args)
    elif args.exp_name == "torre_preprocess":
        create_yaml_torre(args)
