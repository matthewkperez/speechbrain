'''
Update data paths for great lakes
'''

import pandas as pd
from tqdm import tqdm

ROOT_DIR="/home/mkperez/speechbrain/Multilingual"
DATA_DIR=f"{ROOT_DIR}/data"

def data_paths():
    for lang in tqdm(['en', 'fr', 'it','multi','sp']):
        for stype in ['train', 'dev', 'test']:
            src_csv = f"{DATA_DIR}/{lang}/{stype}.csv"
            df = pd.read_csv(src_csv)

            df['wav'] = [w.replace('/home/mkperez/scratch','/home/mkperez') for w in df['wav'].tolist()]
            

            df.to_csv(src_csv)
            # exit()
if __name__ == "__main__":
    data_paths()