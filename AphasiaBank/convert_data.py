'''
Prepare dataset files (csvs)
Convert standard kaldi dir to speechbrain dir

entries include:
utt_ID, duration, wav_path, speaker_ID, transcription
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

def segment_data(seg_file,target_dir,stype,seg_wav_bool):
    '''
    segments wavs and store new wav_path
    return id2wav which has {utt_id: (wav_path,dur,HC_AB)}
    '''
    seg_wav_dir = f"{target_dir}/wavs"
    if not os.path.exists(seg_wav_dir):
        os.makedirs(seg_wav_dir)

    # owav
    control_wav_scp="/z/mkperez/AphasiaBank/kaldi_data_updated/Control/owav.scp"
    aph_wav_scp="/z/mkperez/AphasiaBank/kaldi_data_updated/Aphasia/owav.scp"

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


            if seg_wav_bool:
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



def prepare_duc_data():
    duc_src_dir="/z/mkperez/AphasiaBank/kd_updated/Aphasia+Control/ASR_fold_5"
    target_dir="/z/mkperez/speechbrain/AphasiaBank/data/Duc_process/revised"
    seg_wav_bool = False

    # scores
    scores_xlsx_path = "/z/public/data/AphasiaBank/spkr_info/updated_scores.xlsx"
    df_scores = pd.read_excel(scores_xlsx_path, sheet_name='Time 1')
    df_scores_r = pd.read_excel(scores_xlsx_path, sheet_name='Repeats')
    spk2aq_str = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2aq_str2 = {row['Participant ID']:row['WAB AQ'] for i, row in df_scores_r.iterrows() if row['WAB AQ'] != '0' and row['WAB AQ'] != 'U'}
    spk2type_str = {row['Participant ID']:row['WAB Type'] for i, row in df_scores.iterrows() if row['WAB Type'] != 'U'}
    spk2type_str2 = {row['Participant ID']:row['WAB Type'] for i, row in df_scores_r.iterrows() if row['WAB Type'] != 'U'}
    spk2aq_str.update(spk2aq_str2)
    spk2type_str.update(spk2type_str2)


    # make target directory
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    
    # make train, dev, test csvs
    for stype in ['dev','train', 'test']:
        csv_path = f"{target_dir}/{stype}.csv"
        csv_file = open(csv_path, 'w')
        csv_writer = csv.writer(csv_file)
        # headerdata = ['wrd', 'dataset', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','severity_cat','group','speaker']
        headerdata = ['wrd', 'HC_AB', 'spk_id', 'ID', 'wav', 'duration','severity','aphasia_type','group','severity_cat']
        csv_writer.writerow(headerdata)

        seg_file = f"{duc_src_dir}/{stype}/stored_segments"
        text_file = f"{duc_src_dir}/{stype}/text"
        id2wav = segment_data(seg_file,target_dir,stype,seg_wav_bool)
        id2text = read_text_data(text_file)

        # contains both control and aph speakers
        for utt_id in id2wav.keys():
            group = re.split(r'(\d+)', utt_id)[0]
            speaker = utt_id.split("-")[0]



            # wrd,dataset,spk_id,ID,wav,duration,severity,aphasia_type,severity_cat,group,severity_cat_num
            meta_data = [id2text[utt_id],id2wav[utt_id][2],speaker,utt_id,id2wav[utt_id][0],round(id2wav[utt_id][1],2)]
            if id2wav[utt_id][2] == "Aphasia":
                if speaker not in spk2aq_str:
                    aq = 'N/A'
                    sev_cat = -1
                else:
                    aq = round(spk2aq_str[speaker],2)
                    sev_cat = pd.cut([spk2aq_str[speaker]], bins=[0,25,50,75,100], labels=[4,3,2,1])[0]
                
                if speaker not in spk2type_str:
                    subtype = 'N/A'
                else:
                    subtype=spk2type_str[speaker]
                label_data = [aq,subtype,group,sev_cat]
            else:
                label_data = ['Control' for i in range(3)]
                label_data+=[0]

            csv_data = meta_data + label_data
            csv_writer.writerow(csv_data)

            


if __name__ == "__main__":


    prepare_duc_data()



