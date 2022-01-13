# Multimodal-Clustering-Network
ICCV 2021

This repo has the implementation of our ICCV 2021 paper: Multimodal Clustering Networks for Self-supervised Learning from Unlabeled Videos https://arxiv.org/abs/2104.12671.



Command for pretraining:

```

model1=MCN_sep_recon_r

python -u train_tri_kmeans.py --num_thread_reader=74 --epochs=30 --batch_size=128 \
--n_pair=32 --embd_dim=6144 --howto_audio_frames=1000 --min_time=10.0 --random_audio_windows=0  \
--lr=0.0001 --tri_modal=1 --apex_level=1 --kmeans=1 --use_queue=1 --queue_size=20 --fastC=1 --recon=1 --recon_size=1024 \
--features_path=/nobackup/users/kaudhkha/sightsound/data/howto/parsed_videos \
--features_path_audio=/nobackup/projects/public/howto100m/parsed_videos \
--checkpoint_dir=model_me/$model1 >> logs/$model1
```
