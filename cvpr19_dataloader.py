from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.nn.functional import adaptive_max_pool1d
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import re
import random
import torch.nn.functional as F
import json
import librosa
import math

class CVPR19_DataLoader(Dataset):
    """CVPR19 testset loader."""

    def __init__(
            self,
            csv,
            features_path,
            annot_path,
            steps_path,
            audio_path,
            annot_path_time,
            cook_path,
            with_audio,
            we,
            we_dim=300,
            max_words=30,
            features_path_3D=None,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            num_audio_frames=1024,
            zeus=0,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.annot_path = annot_path
        self.steps_path = steps_path
        self.audio_path = audio_path
        self.annot_path_time = annot_path_time
        self.we = we
        self.we_dim = we_dim
        self.max_words = max_words
        self.feature_framerate = feature_framerate
        self.num_audio_frames = num_audio_frames
        self.zeus = zeus
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = features_path
        #if features_path_3D:
        #    self.feature_path['3d'] = features_path_3D
        self.steps = {}
        self.cook_path = cook_path
        self.cook_set = set()
        self.with_audio = with_audio

        file1 = open(cook_path)
        for line in file1:
            data = line.strip()
            self.cook_set.add(data)
        # for task in self.csv['task'].unique():
        #    with open (os.path.join(self.steps_path,str(task)),'r') as f:
        #        self.steps[str(task)] = th.cat([self._words_to_we(self._tokenize_text(line.strip()))[None,:,:] for line in f],dim=0)
        with open(steps_path, "r") as read_file:
            # print("Converting JSON encoded data into Python dictionary")
            step_dict = json.load(read_file)
        for task, y in step_dict.items():
            self.steps[str(task)] = th.cat([self._words_to_we(self._tokenize_text(step))[None, :, :] for step in y],
                                           dim=0)

    def __len__(self):
        return len(self.csv)

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

    def _zero_pad_audio(self, audio, max_frames):
        n_frames = audio.shape[1]
        if n_frames >= max_frames:
            return audio[:, 0:max_frames], int(max_frames)
        else:
            p = max_frames - n_frames
            audio_padded = np.pad(audio, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
            return audio_padded, n_frames

    #"""
    def _get_video(self, feature_path):
        if self.zeus:
            video = th.load(feature_path).float()
        else:
            video = np.load(feature_path)
        return video if self.zeus else th.from_numpy(video).float()
    #"""

    def _get_video_me(self, vid_path, s, e, fps):
        feature_path = {}
        video = {}
        output = {}
        video = np.load(vid_path)
        video = th.from_numpy(video).float()

        output = th.zeros(len(s), video.shape[-1])
        for i in range(len(s)):
            # start = int(s[i] * fps)
            # end = int(e[i] * fps)
            start = int(i * fps)
            end = int((i + 1) * fps)
            slice = video[start:end]

            output[i] = F.normalize(th.max(slice, dim=0)[0], dim=0)

        return output  # th.cat([output[k] for k in output], dim=1)

    def _get_audio_and_text(self, k, mel_spec):
        # n_caption = len(caption['start'])
        # k = n_pair_max
        starts = np.zeros(k)
        ends = np.zeros(k)
        # text = th.zeros(k, self.max_words, self.we_dim)
        audio = [0 for i in range(k)]

        nframes = np.zeros(k)
        # r_ind = np.random.choice(range(n_caption), k, replace=True)
        dur = 4
        for i in range(k):
            # ind = r_ind[i]
            if i < dur:
                start = 0
                end = 2 * dur
            elif i > k - dur:
                start = k - 2 * dur
                end = k
            else:
                start = i - dur
                end = i + dur
            # print('time',start,end)
            audio[i], nframes[i], starts[i], ends[i] = self._get_single_audio_text(start, end, mel_spec)
        # print('nframes',nframes)
        audio = th.cat([i.unsqueeze(0) for i in audio], dim=0)
        return audio, nframes, starts, ends

    def _get_single_audio_text(self, start, end, mel_spec):

        # words = self._tokenize_text(caption['text'][ind])

        frames = librosa.core.time_to_frames([start, end], sr=16000, hop_length=160, n_fft=400)
        # print('frames',frames[0], frames[1])
        if frames[0] < 0:
            frames[0] = 0
        padded_mel_spec, nframes = self._zero_pad_audio(mel_spec[:, frames[0]: frames[1]], self.num_audio_frames)
        return th.from_numpy(
            padded_mel_spec), nframes, start, end  # , nframes#, caption['start'][start], caption['end'][end], self._words_to_we(words)

    def read_assignment(self, T, K, path):
        Y = np.zeros([T, K], dtype=np.uint8)
        with open(path, 'r') as f:
            for line in f:
                step, start, end = line.strip().split(',')
                start = int(math.floor(float(start)))
                end = int(math.ceil(float(end)))
                step = int(step) - 1
                Y[start:end, step] = 1
        return Y

    def __getitem__(self, idx):
        video_id = self.csv['video_id'][idx]
        task = str(self.csv['task'][idx])
        if self.zeus:
            vid_path_2d = os.path.join(self.feature_path['2d'], self.csv['path'][idx].split('.')[0] + '.pth')
            vid_path_3d = os.path.join(self.feature_path['3d'], self.csv['path'][idx].split('.')[0] + '.pth')
        else:
            # vid_path_2d = os.path.join(self.feature_path['2d'], self.csv['path'][idx])
            # vid_path_3d = os.path.join(self.feature_path['3d'], self.csv['path'][idx])
            vid_path_2d = os.path.join(self.feature_path, self.csv['video_id'][idx] + '_2d.npy')
            vid_path_3d = os.path.join(self.feature_path, self.csv['video_id'][idx] + '_3d.npy')

        annot = th.from_numpy(np.load(os.path.join(self.annot_path, task + '_' + video_id + '.npy')))
        T = annot.size()[0]  # number of frames
        # video[frame,2048] -> [1,2048,frame]
        """
        video_2d = adaptive_max_pool1d(video_2d.transpose(1,0)[None,:,:],T).view(-1,T).transpose(1,0)

        s = [i for i in range(T)]
        e = [i+1 for i in range(T)]
        video_3d_r = th.zeros(T, video_3d.shape[-1])
        for i in range(len(s)):
            start = int(s[i] * self.fps['3d'])
            end = int(e[i] * self.fps['3d']) + 1
            slice_v = video_3d[start:end]
            if len(slice_v) < 1:
                print("error")
            else:
                video_3d_r[i] = F.normalize(th.max(slice_v, dim=0)[0], dim=0)
        video_3d = video_3d_r#adaptive_max_pool1d(video_3d.transpose(1,0)[None,:,:],T).view(-1,T).transpose(1,0)
        """
        # video_3d = adaptive_max_pool1d(video_3d.transpose(1,0)[None,:,:],T).view(-1,T).transpose(1,0)
        #

        # """
        # audio
        au_path = os.path.join(self.audio_path, self.csv['video_id'][idx] + '.npz')
        mel_spec = np.load(au_path)['arr_0']
        audio, nframes, starts, ends = self._get_audio_and_text(T, mel_spec)
        #video_2d = self._get_video_me(vid_path_2d, starts, ends, self.fps['2d'])
        #video_3d = self._get_video_me(vid_path_3d, starts, ends, self.fps['3d'])
        video_2d = self._get_video(vid_path_2d)
        video_3d = self._get_video(vid_path_3d)
        annot = th.from_numpy(np.load(os.path.join(self.annot_path, task + '_' + video_id + '.npy')))
        T = annot.size()[0]
        video_2d = adaptive_max_pool1d(video_2d.transpose(1, 0)[None, :, :], T).view(-1, T).transpose(1, 0)
        video_3d = adaptive_max_pool1d(video_3d.transpose(1, 0)[None, :, :], T).view(-1, T).transpose(1, 0)
        #video = th.cat((F.normalize(video_2d, dim=1), F.normalize(video_3d, dim=1)), dim=1)

        video = th.cat((F.normalize(video_2d, dim=1), F.normalize(video_3d, dim=1)), dim=1)
        #video = th.cat(video_2d,video_3d)

        frames = len(video_2d)
        step_num = len(self.steps[task])
        #annot = self.read_assignment(frames,step_num,os.path.join(self.annot_path_time, task + '_' + video_id + '.csv'))
        # print(video.shape)
        if task in self.cook_set:
            iscook = 1
        else:
            iscook = 0
        if not self.with_audio:
            return {'video': video, 'nframes': th.IntTensor(nframes), 'steps': self.steps[task], 'video_id': video_id,
                    'task': task, 'Y': annot, 'cook': iscook}
        else:
            return {'video': video, 'audio': th.FloatTensor(audio.float()), \
                    'nframes': th.IntTensor(nframes), 'steps': self.steps[task], 'video_id': video_id, \
                    'task': task, 'Y': annot, 'cook': iscook}
