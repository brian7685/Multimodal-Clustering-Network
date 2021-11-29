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

class MSRVTT_DataLoader(Dataset):
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
        #"""
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d_pooled']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d_pooled']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d))
        """
        feat_2d = th.from_numpy(self.data[idx]['2d']).float()
        feat_3d = th.from_numpy(self.data[idx]['3d']).float()
        feat_2d = feat_2d[:10]
        feat_3d = feat_3d[:10]
        #feat_2d = F.interpolate(feat_2d.transpose(1, 0).unsqueeze(0), self.max_frames, mode='linear',
        #                        align_corners=True).squeeze(0)
        #feat_3d = F.interpolate(feat_3d.transpose(1, 0).unsqueeze(0), self.max_frames, mode='linear',
        #                        align_corners=True).squeeze(0)

        feat_2d = F.normalize(feat_2d, dim=1)
        feat_3d = F.normalize(feat_3d, dim=1)
        video = th.cat((feat_2d, feat_3d), dim=1)
        #"""
        # load audio and zero pad/truncate if necessary
        audio = self.data[idx]['audio']
        target_length = 1024 * self.num_frames_multiplier
        nframes = audio.numpy().shape[1]
        p = target_length - nframes
        if p > 0:
            audio = np.pad(audio, ((0,0),(0,p)), 'constant', constant_values=(0,0))
        elif p < 0:
            audio = audio[:,0:p]
        audio = th.FloatTensor(audio)

        # choose a caption
        caption=''
        if self.tri_modal:
            caption = self._get_caption(idx)
            
        return {'video': video, 'text': caption, 'video_id': video_id,
                'audio': audio, 'nframes': nframes}


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
