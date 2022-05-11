import os
import pickle
import numpy as np


#a = np.load('./temp_data/v_ApplyEyeMakeup_g01_c01.npz')
#print(a['arr_0'].shape)
#exit()


def generate_ucf101_pickle():
    data_dir = '/nobackup/users/brian27/data/UCF-101_feature/'
    audio_dir = '/nobackup/users/brian27/data/UCF-101_audio/'
    #data_dir = '../Data/'

    feature_list = os.listdir(data_dir)
    #print(feature_list)
    videos = sorted(set([v[:-6] for v in feature_list]))
    print('# Videos', len(videos))

    train_list = open('./data/ucf_trainlist01.txt').readlines()

    #print(videos)
    #v_Basketball_g07_c02_2d.npy
    data = []
    for video_name in videos:
        training = 0
        for tr_vid in train_list:
            if video_name[:-1] in tr_vid:
                training = 1
        try:
            feats_3d = np.load(data_dir + video_name + '3d.npy')
            #print(feats_3d.shape)
            feats_2d = np.load(data_dir.replace('brian27', 'duartek') + video_name + '2d.npy')
            #print(feats_2d.shape)
        except:
            continue
        try:
            audio = np.load(audio_dir + video_name[:-1] + '.npz')
            print(audio.files, audio_dir + video_name + '.npz', audio['arr_0'].shape)
            audio = audio['arr_0']
            has_audio = 1
        except:
            audio = np.zeros((40, 1), dtype=np.float32)
            has_audio = 0

        data.append({'2d': feats_2d,
                     '3d': feats_3d,
                     '2d_pooled': np.mean(feats_2d, 0),
                     '3d_pooled': np.mean(feats_3d, 0),
                     'class': video_name.split('_')[1],
                     'video': video_name,
                     'audio': audio,
                     'has_audio': has_audio,
                     'training': training
        })
    pickle.dump(data, open('./data/UCF101_data.pkl', 'wb'))
    print('# Videos with features extracted:', len(data))
    #a = os.listdir('/nobackup/users/brian27/data/hmdb51_feature/')


def generate_hmdb_pickle():
    data_dir = '/nobackup/users/brian27/data/hmdb51_feature/'
    folders_dir = '/nobackup/users/brian27/data/hmdb51_org/'

    classes = os.listdir(folders_dir)

    feature_list = os.listdir(data_dir)
    videos = sorted(set([v[:-6] for v in feature_list]))
    print('# Videos', len(videos))

    train_list = open('./data/hmdb_train_split1.txt').readlines()
    test_list = open('./data/hmdb_test_split1.txt').readlines()

    n_samples = np.zeros((len(classes), ))
    data = []
    for video_name in videos:
        training = 0
        for tr_vid in train_list:
            if video_name[:-1] in tr_vid:
                training = 1

        testing = 0
        for te_vid in test_list:
            if video_name[:-1] in te_vid:
                testing = 1

        if training == 0 and testing == 0:
            training = 2

        try:
            feats_3d = np.load(data_dir + video_name + '3d.npy')
            #print(feats_3d.shape)
            feats_2d = np.load(data_dir.replace('brian27', 'duartek') + video_name + '2d.npy')
            #print(feats_2d.shape)
        except:
            continue

        split_name = '_'.join(video_name.split('_')[:-7]) + '_'
        class_name = [cls for cls in classes if '_'+cls+'_' == split_name[-(len(cls)+2):]]
        class_name = sorted(class_name, key=lambda x: len(x))
        #print(class_name, class_name[-1])
        class_name = class_name[-1]
        n_samples[classes.index(class_name)] += 1
        data.append({'2d': feats_2d,
                     '3d': feats_3d,
                     '2d_pooled': np.mean(feats_2d, 0),
                     '3d_pooled': np.mean(feats_3d, 0),
                     'class': class_name,
                     'video': video_name,
                     'training': training
        })
    pickle.dump(data, open('./data/HMDB_data.pkl', 'wb'))
    print('# Videos with features extracted:', len(data))
    for i, cls in enumerate(classes):
        print(cls, n_samples[i])
    #print(n_samples)

generate_ucf101_pickle()
generate_hmdb_pickle()
