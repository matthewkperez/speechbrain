


stage=2


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
    echo "Torre"
    # subtype=("freeze_w2v" "last2_w2v")
    subtype=("last2_w2v")
    for index in "${!subtype[@]}"; do
        title=${subtype[${index}]}
        echo ${title}
        exp_name="torre_preprocess"
        out_yaml="/y/mkperez/speechbrain/AphasiaBank/hparams/${exp_name}/${title}.yaml"
        python yaml_helper.py -e ${exp_name} -s ${title} -o ${out_yaml}
        
        if [ $title == "last2_w2v" ]; then
            og_pretrain="results/torre/freeze-True/wav2vec2-large-960h-lv60-self-2000/ep-20_lr-0.9_lr_w2v-0.0001"
            # new_finetune="results/torre/freeze-True/wav2vec2-large-960h-lv60-self-2000"

            lr_w2v="0.001"
            lr="0.9"
            new_finetune="results/torre/freeze-False/wav2vec2-large-960h-lv60-self-2000/lr-${lr}_lr_w2v-${lr_w2v}"
            if [ -d ${new_finetune} ]; then
                echo "remove dir"
                rm -r ${new_finetune}
            fi
            mkdir -p ${new_finetune}

            cp -r ${og_pretrain}/save ${new_finetune}
        fi

        ### launch script
        # CUDA_VISIBLE_DEVICES=0 python selective_training_torre.py ${out_yaml} &>logs/torre/FT_lr-${lr}_lr_w2v-${lr_w2v}.txt &
        CUDA_VISIBLE_DEVICES=0 python selective_training_torre.py ${out_yaml}

        python analyze_wer.py -y ${out_yaml}
    done
fi




if [ $stage == 0 ] || [ $stage == 2 ]; then
    echo "Manual run"

    CUDA_VISIBLE_DEVICES=2 python selective_training_torre.py hparams/manual_run.yaml


fi