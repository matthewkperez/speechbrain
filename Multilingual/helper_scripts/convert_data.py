'''
Prepare dataset files (csvs)
Convert standard kaldi dir to speechbrain dir

entries include:
utt_ID, duration, wav_path, speaker_ID, transcription, lang_id
'''
from lib2to3.pgen2.pgen import DFAState
import os
import pandas as pd
import math
import chaipy.io
import shutil
from tqdm import tqdm
import re
import subprocess
import csv

ROOT_DIR = "/z/mkperez/AphasiaBank/multilingual"
SB_DIR = "/z/mkperez/speechbrain/Multilingual"
SEG_WAV_BOOL = False

DIR2LANG = {'English-Aphasia':'en', 'English-Control':'en', 'French':'fr', 'Italian':'it', 'Spanish':'sp'}
LANG2DIR = {'en': ['English-Aphasia', 'English-Control'], 'fr': ['French'], 'it':['Italian'], 'sp':['Spanish'], 'multi':['English-Aphasia', 'English-Control', 'French', 'Italian', 'Spanish']}

def segment_data(lang, seg_file,target_dir,stype):
    '''
    segments wavs and store new wav_path
    return id2wav which has {utt_id: (wav_path,dur,HC_AB)}
    '''
    seg_wav_dir = f"{target_dir}/../wavs"
    if not os.path.exists(seg_wav_dir):
        os.makedirs(seg_wav_dir)

    # owav
    control_wav_scp=f"{ROOT_DIR}/{lang}/owav.scp"
    aph_wav_scp=f"{ROOT_DIR}/{lang}/owav.scp"

    # create spkr->src wav dict
    spkr_src_wav_dict={}
    HC_AB_dict = {}
    for scp, AB_HC in zip([control_wav_scp, aph_wav_scp], ['Control', 'Aphasia']):
        with open(scp, 'r') as r:
            lines = r.readlines()
            for line in tqdm(lines, total=len(lines)):
                spkr_id = line.split()[0]
                wav_path = line.split()[1]
                spkr_src_wav_dict[spkr_id]=wav_path
                HC_AB_dict[spkr_id] = AB_HC


    missed_files=[]
    id2wav = {}
    with open(seg_file, 'r') as r:
        # get control/aphasia and speaker
        lines = r.readlines()
        for line in tqdm(lines):
            utt_id = line.split()[0]
            speaker = line.split()[1]
            start_t = float(line.split()[2])
            end_t = float(line.split()[3])
            dur = end_t-start_t


            src_wav_path = spkr_src_wav_dict[speaker]

            new_seg_wav_path = f"{seg_wav_dir}/{utt_id}.wav"


            if SEG_WAV_BOOL:
                # sox segmentation
                cmd = ['sox', src_wav_path, new_seg_wav_path, 'trim', f"{start_t}", f"={end_t}"]
                list_files = subprocess.run(cmd)
                # assert list_files.returncode == 0, f"sox error {utt_id}: {start_s} = {end_s}"
                if list_files.returncode != 0:
                    missed_files.append(utt_id)
                    continue
            


            id2wav[utt_id] = (new_seg_wav_path,dur,HC_AB_dict[speaker])

    # output missed files
    if len(missed_files) > 1:
        with open(f'missed_files_{stype}.txt', 'w') as w:
            for line in missed_files:
                w.write(f"line\n")

    return id2wav

def norm_text(text):
    text = text.lower().strip()
    return text

def read_text_data(text_file):
    '''
    output id2text = {utt_id: text}
    '''
    id2text={}
    with open(text_file, 'r') as r:
        lines = r.readlines()
        for line in tqdm(lines):
            id = line.split()[0]
            text = line.split(" ", 1)[1]

            id2text[id] = norm_text(text)
            # print(f"pre: {text}")
            # print(f"norm: {id2text[id]}")
            # exit()

    return id2text

def prepare_data(lang_key):
    target_dir=f"{SB_DIR}/data/{lang_key}"
    # make target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # make train, dev, test csvs
    for stype in ['dev','train', 'test']:
        # CSV init
        csv_path = f"{target_dir}/{stype}.csv"
        csv_file = open(csv_path, 'w')
        csv_writer = csv.writer(csv_file)
        headerdata = ['wrd', 'lang_id', 'ID', 'wav', 'duration']
        csv_writer.writerow(headerdata)


        lang_dirs = LANG2DIR[lang_key]
        lang_id_tags = [DIR2LANG[l] for l in lang_dirs]


        for lang, lang_id in zip(lang_dirs, lang_id_tags):
            src_dir = f"{ROOT_DIR}/{lang}/CV/Fold_1"

            # add all English-Control to train set
            if stype == "train" and lang == "English-Control":
                id2wav = {}
                id2text = {}
                for stype_en_control in ['dev','train', 'test']:
                    # add all to train set
                    seg_file = f"{src_dir}/{stype_en_control}/segments"
                    text_file = f"{src_dir}/{stype_en_control}/text"
                    id2wav.update(segment_data(lang, seg_file,target_dir,stype))
                    id2text.update(read_text_data(text_file))
            else:

                # read data
                seg_file = f"{src_dir}/{stype}/segments"
                text_file = f"{src_dir}/{stype}/text"
                id2wav = segment_data(lang, seg_file,target_dir,stype)
                id2text = read_text_data(text_file)

            # Write data
            for utt_id in id2wav.keys():
                # ['wrd', 'lang_id', 'ID', 'wav', 'duration']
                meta_data = [id2text[utt_id], lang_id, utt_id, id2wav[utt_id][0], round(id2wav[utt_id][1],2)]
                csv_data = meta_data
                csv_writer.writerow(csv_data)

            


if __name__ == "__main__":

    for lang_id in ['en','fr','it','sp', 'multi']:
        prepare_data(lang_key=lang_id)



