#!/bin/bash
#SBATCH --qos=sched_level_2
#SBATCH --gres=gpu:4 
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --time=2:00:00
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

#python gen_loader.py

#python eval.py --eval_youcook=1 --num_thread_reader=74 --embd_dim=6144 --pretrain_path=/nobackup/users/brian27/howto100m/model/howto100m_pt_model.pth

python eval.py --eval_youcook=1 --num_thread_reader=74 --embd_dim=4096 --pretrain_path=/nobackup/users/brian27/howto100m/model_me/mil_nce_two/e18.pth


#python train_tri_kmeans.py --eval_youcook=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e16.pth

#python train_tri_kmeans.py --eval_msrvtt=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e16.pth


echo "Weights 16"

#python train_tri_kmeans.py --eval_ucf=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e16.pth

python train_tri_kmeans.py --eval_hmdb=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
--pretrain_path=model_mcn/MCN_KMeans/e16.pth



#echo "Weights 21"

#python train_tri_kmeans.py --eval_ucf=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e21.pth

#python train_tri_kmeans.py --eval_hmdb=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e21.pth

#echo "Weights 24"

#python train_tri_kmeans.py --eval_ucf=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e24.pth

#python train_tri_kmeans.py --eval_hmdb=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e24.pth


echo "Weights 26"

#python train_tri_kmeans.py --eval_ucf=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e26.pth

python train_tri_kmeans.py --eval_hmdb=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
--pretrain_path=model_mcn/MCN_KMeans/e26.pth





#python train_tri_kmeans.py --eval_cross=1 --num_thread_reader=74 --epochs=0 --batch_size=512 \
#--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
#--lr=0.0001 --tri_modal=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --mean=1 --recon=0 --recon_size=1024 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
#--pretrain_path=model_mcn/MCN_KMeans/e16.pth

#python train_tri_cos_mil.py --eval_cross=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Joint_Recon_Hard/e15.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

#python train_tri_cos_mil.py --eval_youcook=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Joint_Recon_Cross_Hard/e9.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

#python train_tri_cos_mil.py --eval_msrvtt=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Joint_Recon_Cross_Hard/e9.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

#python train_tri_cos_mil.py --eval_ucf=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Sports/e20.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

#python train_tri_cos_mil.py --eval_hmdb=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Sports/e20.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

#python train_tri_cos_mil.py --eval_msrvtt=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#python train_tri_cos_mil.py --eval_youcook=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#python train_tri_cos_mil.py --eval_ucf=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#python train_tri_cos_mil.py --eval_hmdb=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Joint_Recon/e11.pth \
#--lr=1e-5 --tri_modal=1 --finetune_video=0 --video_interp=0 --output_norm=1 --joint_cluster=1 --multi_head=0 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos


#python local_eval.py

# model_mcn/MCN1/e9.pth

#python train_tri_c.py --eval_youcook=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Recon2/e10.pth \
#--lr=1e-5 --tri_modal=1



#python train_tri_c.py --eval_msrvtt=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Recon2/e14.pth \
#--lr=1e-5 --tri_modal=1 \
#--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
#--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos

# model_mcn/MCN1/e9.pth

#python train_tri_c.py --eval_youcook=1  --num_thread_reader=74 --batch_size=512 --epochs=0 --project=1 --project_dim=8000 \
#--lr_decay=1.0 --embd_dim=6144  --pretrain_path=model_mcn/MCN_Recon2/e14.pth \
#--lr=1e-5 --tri_modal=1










## Wait for all commands to finish
wait 
echo "Run completed at:- "
date
