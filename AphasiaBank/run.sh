


stage=6


if [ $stage == 0 ] || [ $stage == 1 ]; then
    echo "severity training"
    # severity_arr=("tier-severe" "tier-moderate" "tier-mild" "baseline")
    severity_arr=("baseline")
    for index in "${!severity_arr[@]}"; do
        sev_title=${severity_arr[${index}]}
        echo ${sev_title}
        exp_name="severity_training"
        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/${exp_name}/${sev_title}.yaml"
        python yaml_helper.py -e ${exp_name} -s ${sev_title} -o ${out_yaml}

        ### launch script
        # CUDA_VISIBLE_DEVICES=2 python selective_training.py ${out_yaml} &>logs/${sev_title}.txt &
        CUDA_VISIBLE_DEVICES=0 python selective_training.py ${out_yaml}

        ### launch script DP (cannot use AMP)
        ## https://forum.ailab.unb.br/t/data-parallel-dp-e-distributed-data-parallel-ddp-training-in-pytorch-e-fastai-v2/194
        # python selective_training.py ${out_yaml} --data_parallel_backend

        ### launch script DDP
        # python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 --node_rank=0 selective_training.py ${out_yaml} --distributed_launch --distributed_backend='nccl' --find_unused_parameters
        # python -m torch.distributed.launch --nproc_per_node=3 --nnodes=1 selective_training.py ${out_yaml} --distributed_launch --distributed_backend='nccl'

        # exit
    done
fi

if [ $stage == 0 ] || [ $stage == 2 ]; then
    echo "Pretrain and Finetune"
    # subtype=("freeze_w2v" "last2_w2v")
    subtype=("last2_w2v")
    for index in "${!subtype[@]}"; do
        title=${subtype[${index}]}
        echo ${title}
        # exp_name="torre_preprocess"
        exp_name="personalization"
        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/${exp_name}/${title}.yaml"
        python yaml_helper.py -e ${exp_name} -s ${title} -o ${out_yaml}
        
        if [ $title == "last2_w2v" ]; then
            tr="train_all" # "train_no_kansas" = pts-12, "train_all" = pts-11
            val="val_no_kansas"
            og_pretrain="results/${exp_name}/freeze-True/tr-${tr}_val-${val}"
            new_finetune="results/${exp_name}/freeze-False/tr-${tr}_val-${val}"
            if [ -d ${new_finetune} ]; then
                echo "remove dir"
                rm -r ${new_finetune}
            fi
            mkdir -p ${new_finetune}
            cp -r ${og_pretrain}/save ${new_finetune}
        
        elif [ $title == "personalize" ]; then
            tr="train_no_kansas" # "train_no_kansas" = pts-12, "train_all" = pts-11
            val="val_no_kansas"
            og_pretrain="results/${exp_name}/freeze-False/tr-${tr}_val-${val}"
            new_finetune="results/${exp_name}/freeze-False/tr-train_kansas_val-val_kansas"
            if [ -d ${new_finetune} ]; then
                echo "remove dir"
                rm -r ${new_finetune}
            fi
            mkdir -p ${new_finetune}
            cp -r ${og_pretrain}/save ${new_finetune}
        
        fi

        # exit
        ### launch script
        # CUDA_VISIBLE_DEVICES=0 python selective_training_torre.py ${out_yaml} &>logs/torre/FT_lr-${lr}_lr_w2v-${lr_w2v}.txt &
        CUDA_VISIBLE_DEVICES=2 python selective_training_torre.py ${out_yaml}

        # python analyze_wer.py -y ${out_yaml}
    done
fi

if [ $stage == 0 ] || [ $stage == 3 ]; then
    echo "Personalize"
    kansas_wav_ar=(/y/public/data/AphasiaBank/Aphasia/Kansas/*.wav)
    for index in "${!kansas_wav_ar[@]}"; do
        title="personalize"
        kansas_speaker_wav=`basename ${kansas_wav_ar[$index]}`
        kansas_speaker=${kansas_speaker_wav%.wav}

        exp_name="personalization"
        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/${exp_name}/${title}.yaml"
        python yaml_helper.py -e ${exp_name} -s ${title} -o ${out_yaml} -p $kansas_speaker
    
        if [ $title == "personalize" ]; then
            tr="train_all" # "train_no_kansas" = pts-12, "train_all" = pts-11
            val="val_no_kansas"
            og_pretrain="results/${exp_name}/freeze-False/tr-${tr}_val-${val}"
            new_finetune="results/${exp_name}/freeze-False/SD-${kansas_speaker}"
            if [ -d ${new_finetune} ]; then
                echo "remove dir"
                rm -r ${new_finetune}
            fi
            mkdir -p ${new_finetune}
            cp -r ${og_pretrain}/save ${new_finetune}
        
        fi
        # exit

        ### launch script
        # CUDA_VISIBLE_DEVICES=0 python selective_training_torre.py ${out_yaml} &>logs/torre/FT_lr-${lr}_lr_w2v-${lr_w2v}.txt &
        CUDA_VISIBLE_DEVICES=2 python selective_training_torre.py ${out_yaml}
        # exit
        # python analyze_wer.py -y ${out_yaml}
    done
fi




if [ $stage == 0 ] || [ $stage == 4 ]; then
    echo "Manual run"

    stages="warmup FT" # warmup FT
    for stage in $stages; do
        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/fluency/updated.yaml"
        python yaml_helper.py -e "PT_FT" -s $stage -o $out_yaml

    done

    CUDA_VISIBLE_DEVICES=2 python train.py hparams/Duc_process/base.yaml
fi

if [ $stage == 0 ] || [ $stage == 5 ]; then
    echo "Pretrain and Finetune based on fluency"
    data_dir="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process"
    
    # # # csv generation
    # python helper_scripts/partition_PT-FT_data.py 'fluency' $data_dir

    # PT_FT_list="PT FT"
    fluent="Fluent Non-Fluent" # Fluent Non-Fluent
    PT_FT_list="PT" # PT FT
    for PT_FT_var in $PT_FT_list; do
        if [ $PT_FT_var == 'FT' ]; then
            PT_path="/y/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_fluency/PT-${fluent}/No-LM_wav2vec2/freeze-False"
            FT_path="/y/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_fluency/FT-${fluent}/No-LM_wav2vec2/freeze-False"
            if [ ! -d FT_path ]; then
                mkdir -p $FT_path
                cp -r $PT_path/save $FT_path
            fi
        fi

        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/fluency/updated.yaml"
        python yaml_helper.py -e "PT_FT" -s ${PT_FT_var} -o ${out_yaml} -f ${fluent} 
        
        # exit
        CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py $out_yaml
    done

    
fi

if [ $stage == 0 ] || [ $stage == 6 ]; then
    echo "Pretrain and Finetune based on subtype"
    data_dir="/y/mkperez/speechbrain/AphasiaBank/data/Duc_process"
    
    # # # csv generation
    # python helper_scripts/partition_PT-FT_data.py 'subtype' $data_dir
    # exit

    # subtypes="Anomic Broca Conduction Global TransMotor TransSensory Wernicke" # Fluent Non-Fluent
    # subtypes="Anomic Broca Conduction Wernicke" 
    subtypes="Anomic" 
    PT_FT_list="PT FT" # PT FT
    for subtype in $subtypes; do
        for PT_FT_var in $PT_FT_list; do
            if [ $PT_FT_var == 'FT' ]; then
                PT_path="/y/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_subtype/PT-${subtype}/No-LM_wav2vec2/freeze-False"
                FT_path="/y/mkperez/speechbrain/AphasiaBank/results/duc_process/PT-FT_subtype/FT-${subtype}/No-LM_wav2vec2/freeze-False"
                if [ ! -d FT_path ]; then
                    mkdir -p $FT_path
                    cp -r $PT_path/save $FT_path
                fi
            fi

            out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/Duc_process/PT_FT/subtype/updated.yaml"
            python yaml_helper.py -e "PT_FT" -s ${PT_FT_var} -o ${out_yaml} -t ${subtype} 
            
      
            # CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 train.py $out_yaml --distributed_launch --distributed_backend='nccl' --find_unused_parameters
            CUDA_VISIBLE_DEVICES=0,1,2 python train.py $out_yaml --data_parallel_backend

            exit
        done
    done
    
fi