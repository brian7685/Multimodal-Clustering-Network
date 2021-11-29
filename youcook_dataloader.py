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
from torch.utils.data.dataloader import default_collate

class Youcook_DataLoader(Dataset):
    """Youcook dataset loader."""

    def __init__(
            self,
            data,
            we,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5,
            tri_modal=False,    
    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb'))
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_frames_multiplier = num_frames_multiplier
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

    def __getitem__(self, idx):
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
        video = th.cat((feat_2d, feat_3d))

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

        caption = ''
        if self.tri_modal:
            caption = self._words_to_we(self._tokenize_text(self.data[idx]['caption'])) 

        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': audio, 'nframes': nframes}


class Youcook_DataLoader_label(Dataset):
    """Youcook dataset loader."""

    def __init__(
            self,
            data,
            we,
            pseudo_v,
            pseudo_a,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5,
            tri_modal=False,

    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb')) #9000*4800
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_frames_multiplier = num_frames_multiplier
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

    def __getitem__(self, idx):
        # load 2d and 3d features (features are pooled over the time dimension)
        feat_2d = F.normalize(th.from_numpy(self.data[idx]['2d']).float(), dim=0)
        feat_3d = F.normalize(th.from_numpy(self.data[idx]['3d']).float(), dim=0)
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

        caption = ''
        if self.tri_modal:
            caption = self._words_to_we(self._tokenize_text(self.data[idx]['caption']))

        return {'video': video, 'text': caption, 'video_id': self.data[idx]['id'],
                'audio': audio, 'nframes': nframes, 'pseudo_v':self.pseudo_v[idx], 'pseudo_a':self.pseudo_a[idx]}

class Youcook_DataLoader_knn(Dataset):
    """Youcook dataset loader."""

    def __init__(
            self,
            data,
            we,
            knn_v,
            knn_a,
            we_dim=300,
            max_words=30,
            num_frames_multiplier=5,
            tri_modal=False,

    ):
        """
        Args:
        """
        self.data = pickle.load(open(data, 'rb')) #9000*4800
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.num_frames_multiplier = num_frames_multiplier
        self.tri_modal = tri_modal
        self.knn_v = knn_v
        self.knn_a = knn_a

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

    def __getitem__(self, idx):
        video_feature = []
        text_feature = []
        audio_feature = []
        nframes_list = []
        caption_text = []
        for i in self.knn_v[idx]:
            # load 2d and 3d features (features are pooled over the time dimension)
            feat_2d = F.normalize(th.from_numpy(self.data[i]['2d']).float(), dim=0)
            feat_3d = F.normalize(th.from_numpy(self.data[i]['3d']).float(), dim=0)
            video = th.cat((feat_2d, feat_3d))
            video_feature.append(video.numpy())
            # load audio and zero pad/truncate if necessary
            audio = self.data[i]['audio']
            target_length = 1024 * self.num_frames_multiplier
            nframes = audio.numpy().shape[1]
            nframes_list.append(nframes)
            p = target_length - nframes
            if p > 0:
                audio = np.pad(audio, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
            elif p < 0:
                audio = audio[:, 0:p]
            audio = th.FloatTensor(audio)
            audio_feature.append(audio.numpy())
            caption = ''
            if self.tri_modal:
                caption = self._words_to_we(self._tokenize_text(self.data[i]['caption']))
                text_feature.append(caption.numpy())
        video_f = np.asarray(video_feature)
        text_f = np.asarray(text_feature)
        audio_f = np.asarray(audio_feature)
        nframes_l = np.asarray(nframes_list)
        """
        print('dataload')
        print(video_f.shape)
        print(text_f.shape)
        print(audio_f.shape)
        print(nframes_l.shape)
        print('dataload_fin')
        """
        #caption_text =
        return {'video': video_f, 'text': text_f, 'video_id': self.data[i]['id'],
                'audio': audio_f, 'nframes': nframes_l}