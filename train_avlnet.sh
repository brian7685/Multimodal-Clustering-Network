#!/bin/bash
#SBATCH --qos=sched_level_2
#SBATCH --gres=gpu:4 
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task 74
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1T
#SBATCH --exclusive
#SBATCH --job-name="ht"
#SBATCH --output logs/ht-%j.out
#SBATCH --error logs/ht-%j.err
## NOTE: adjust the dependency if needed for the 2nd and 3rd run
##SBATCH --dependency=afterok:12625

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " GPUs per node:= " $SLURM_JOB_GPUS
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes/GPU devices"
echo ""
echo " Run started at:- "
date

source /nobackup/users/duartek/anaconda3/bin/activate
conda activate wmlce-1.6.2

nvidia-smi
pwd

#####################

 
python -u train_tri_kmeans.py --num_thread_reader=74 --epochs=10 --batch_size=128 \
--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
--lr=0.0001 --tri_modal=1 --apex_level=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=1 --recon_size=1024 \
--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
--pretrain_path=model_mcn/MCN_KMeans/e16.pth --train_csv=data/HowTo100M_336_videopaths.txt \
--checkpoint_dir=model_mcn/MCN_KMeans >> logs/MCN_KMeans


#python -u train_tri_cos_mil.py --num_thread_reader=74 --epochs=30 --batch_size=128 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0 --finetune_video=0 --video_interp=0 \
#--recon=1 --recon_b=0 --recon_cross=0 --joint_cluster=1 --cluster_a=0 --multi_head=0 \
#--lr=0.0001 --tri_modal=1 --apex_level=1 --cluster=1 --soft_label=0 --start_cluster=0 --project=1 --project_dim=8000 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_Sports/e10.pth --train_csv=data/HowTo100M_336_videopaths.txt \
#--checkpoint_dir=model_mcn/MCN_Sports >> logs/MCN_Sports

# --pretrain_path=/nobackup/users/brian27/MCN_public/model_mcn/$model1/e9.pth \
## Run two training commands in the background, each on two V100 GPUs
#model1=AVLnet_test_code_release
#model2=AVLnet_text_test_code_release


#CUDA_VISIBLE_DEVICES=0,1 python -u train.py --num_thread_reader=20 --epochs=7 --batch_size=128 --n_pair=32 --embd_dim=4096 --howto_audio_frames=1000 --lr=0.001 --apex_level=1 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos --features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--checkpoint_dir=model/$model1 >> logs/$model1 & \

## Add --pretrain_path to the command before the >> for the second run
# --pretrain_path=model/$model1/e7.pth

#CUDA_VISIBLE_DEVICES=2,3 python -u train.py --num_thread_reader=20 --epochs=7 --batch_size=128 --n_pair=32 --embd_dim=4096 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --tri_modal_fuse=1 --apex_level=1 --features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos --checkpoint_dir=model/$model2 >> logs/$model2 & \

## Add --pretrain_path to the command before the >> for the second run 
# --pretrain_path=model/$model2/e7.pth

## Wait for all commands to finish
wait 
echo "Run completed at:- "
date
