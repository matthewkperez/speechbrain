




stage=2



if [ $stage == 0 ]; then
    echo "create .txt and .json files for LM.yaml"
    python prep_LM.py -d "data/Duc_process"

fi

if [ $stage == 1 ]; then
    echo "Train tokenizer"
    python /z/mkperez/speechbrain/templates/speech_recognition/Tokenizer/train.py "hparams/Duc_process/tokenizer.yaml"
fi

if [ $stage == 2 ]; then
    echo "train model"
    # rm -r "/y/mkperez/speechbrain/AphasiaBank/data/Duc_process/RNNLM"
    CUDA_VISIBLE_DEVICES=2 python /z/mkperez/speechbrain/templates/speech_recognition/LM/train.py "hparams/Duc_process/LM.yaml"

fi

if [ $stage == 3 ]; then
    echo "symlink lm.ckpt and tokenizer.ckpt to a folder"
    lm_path="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process/RNNLM/save/CKPT+2023-02-05+21-45-37+00"
    tok_path="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process/tokenizer_1k/1000_unigram.model"

    ln -s $tok_path $lm_path

fi