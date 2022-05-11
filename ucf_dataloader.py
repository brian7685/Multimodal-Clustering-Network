from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import pickle
import torch.nn.functional as F
import numpy as np
import re
import pandas as pd
from collections import defaultdict
from torch.utils.data.dataloader import default_collate
import json
import random


def name_to_stringlist(name):
    change = {}
    """
    change = {'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups']}
    """
    """
    change = {
        'CleanAndJerk': ['weight', 'lift'],
        'Skijet': ['Skyjet'],
        'HandStandPushups': ['handstand', 'pushups'],
        'HandstandPushups': ['handstand', 'pushups'],
        'PushUps': ['pushups'],
        'PullUps': ['pullups'],
        'WalkingWithDog': ['walk', 'dog'],
        'ThrowDiscus': ['throw', 'disc'],
        'TaiChi': ['taichi'],
        'CuttingInKitchen': ['cut', 'kitchen'],
        'YoYo': ['yoyo'],
    }
    """
    if name in change:
        name_vec = change[name]
    else:
        upper_idx = np.where([x.isupper() for x in name])[0].tolist()
        upper_idx += [len(name)]
        name_vec = []
        for i in range(len(upper_idx)-1):
            name_vec.append(name[upper_idx[i]: upper_idx[i+1]])
        name_vec = [n.lower() for n in name_vec]
        #name_vec = verbs2basicform(name_vec)
    return name_vec


class UCF_DataLoader(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            data_path,
            we,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5,
            training=True,
            tri_modal=False,
            finetune_video=False,
            video_interp=False
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data_path, 'rb'))  # contains a list of video names
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.max_video = 30
        self.num_frames_multiplier = num_frames_multiplier
        self.training = training
        self.tri_modal = tri_modal
        self.finetune_video = finetune_video
        self.max_frames = 16
        self.video_interp = video_interp

        names = []
        for vid in self.data:
            names.append(vid['class'])

        self.classes = sorted(set(names))
        print('# Classes', len(self.classes))

        self.class_embeds = []
        for name in self.classes:
            word_list = name_to_stringlist(name)
            caption = ' '.join(word_list)
            self.class_embeds.append(self._get_caption(caption))
        self.class_embeds = th.stack(self.class_embeds, 0)
        print('Shape of class embeds', self.class_embeds.shape)

    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def _get_caption(self, idx):
        """Chooses random caption if training. Uses set caption if evaluating."""
        if self.training:
            captions = idx
            caption = self._words_to_we(self._tokenize_text(random.choice(captions)))
            return caption
        else:
            caption = idx
            return self._words_to_we(self._tokenize_text(caption))

    def __getitem__(self, idx):
        data = self.data[idx]
        # load 2d and 3d features (features are pooled over the time dimension)

        if self.finetune_video:
            feat_2d = th.from_numpy(self.data[idx]['2d']).float()
            feat_3d = th.from_numpy(self.data[idx]['3d']).float()
            if self.video_interp:
                feat_2d = F.interpolate(feat_2d.transpose(1, 0).unsqueeze(0), self.max_frames, mode='linear',
                                        align_corners=True).squeeze(0)
                feat_3d = F.interpolate(feat_3d.transpose(1, 0).unsqueeze(0), self.max_frames, mode='linear',
                                        align_corners=True).squeeze(0)
            else:
                feat2d_buffer = th.zeros(self.max_frames, feat_2d.shape[-1])
                feat_2d = feat_2d[:self.max_frames]
                feat2d_buffer[:len(feat_2d)] = feat_2d

                feat3d_buffer = th.zeros(self.max_frames, feat_3d.shape[-1])
                feat_3d = feat_3d[:self.max_frames]
                feat3d_buffer[:len(feat_3d)] = feat_3d

                feat_2d = feat2d_buffer.transpose(1, 0)
                feat_3d = feat3d_buffer.transpose(1, 0)

            feat_2d = F.normalize(feat_2d, dim=0)
            feat_3d = F.normalize(feat_3d, dim=0)
            video = th.cat((feat_2d, feat_3d), dim=0)
        else:
            feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d_pooled']).float(), dim=0)
            feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d_pooled']).float(), dim=0)
            video = th.cat((feat_2d, feat_3d))

        # load audio and zero pad/truncate if necessary
        audio = self.data[idx]['audio']
        target_length = 1024 * self.num_frames_multiplier
        nframes = audio.shape[1]
        p = target_length - nframes
        if p > 0:
            audio = np.pad(audio, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
        elif p < 0:
            audio = audio[:, 0:p]
        audio = th.FloatTensor(audio)

        # choose a caption
        caption = ''
        name = self.data[idx]['class']
        if self.tri_modal:
            word_list = name_to_stringlist(name)
            caption = ' '.join(word_list)
            caption = self._get_caption(caption)

        return {'video': video, 'text': caption, 'video_id': idx,
                'audio': audio, 'nframes': 32, 'class_name': name,
                'class_id': th.ones(1)*self.classes.index(name),
                'has_audio': th.ones(1)*self.data[idx]['has_audio'],
                'video_name': self.data[idx]['video'],
                'training': th.ones(1)*self.data[idx]['training']}


class MSRVTT_DataLoader_label(Dataset):
    """MSRVTT dataset loader."""

    def __init__(
            self,
            data_path,
            we,
            pseudo_v,
            pseudo_a,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5,
            training=True,
            tri_modal=False,
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data_path, 'rb'))
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.max_video = 30
        self.num_frames_multiplier = num_frames_multiplier
        self.training = training
        self.tri_modal = tri_modal
        self.pseudo_v = pseudo_v
        self.pseudo_a = pseudo_a



    def __len__(self):
        return len(self.data)

    def custom_collate(self, batch):
        return default_collate(batch)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _tokenize_text(self, sentence):
        w = re.findall(r"[\w']+", str(sentence))
        return w

    def _words_to_we(self, words):
        words = [word for word in words if word in self.we.vocab]
        if words:
            we = self._zero_pad_tensor(self.we[words], self.max_words)
            return th.from_numpy(we)
        else:
            return th.zeros(self.max_words, self.we_dim)

    def _get_caption(self, idx):
        """Chooses random caption if training. Uses set caption if evaluating."""
        if self.training:
            captions = self.data[idx]['caption']
            caption = self._words_to_we(self._tokenize_text(random.choice(captions)))
            return caption
        else:
            caption = self.data[idx]['eval_caption']
            return self._words_to_we(self._tokenize_text(caption))

    def __getitem__(self, idx):
        video_id = self.data[idx]['id']
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d_pooled']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d_pooled']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d))

        # load audio and zero pad/truncate if necessary
        audio = self.data[idx]['audio']
        target_length = 1024 * self.num_frames_multiplier
        nframes = audio.numpy().shape[1]
        p = target_length - nframes
        if p > 0:
            audio = np.pad(audio, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
        elif p < 0:
            audio = audio[:, 0:p]
        audio = th.FloatTensor(audio)

        # choose a caption
        caption = ''
        if self.tri_modal:
            caption = self._get_caption(idx)

        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': audio, 'nframes': nframes, 'pseudo_v': self.pseudo_v[idx], 'pseudo_a': self.pseudo_a[idx]}
