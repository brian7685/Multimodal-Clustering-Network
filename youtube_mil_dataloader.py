from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import Dataset
import torch.nn.functional as F
import pandas as pd
import os
import numpy as np
import re
import random
import librosa
from model_davenet import LoadAudio


class Youtube_DataLoader(Dataset):
    """Youtube dataset loader."""

    def __init__(
            self,
            csv,
            features_path,
            features_path_audio,
            caption,
            we,
            min_time=10.0,
            feature_framerate=1.0,
            feature_framerate_3D=24.0 / 16.0,
            we_dim=300,
            max_words=30,
            min_words=0,
            n_pair=1,
            num_audio_frames=1024,
            num_candidates=1,
            random_audio_windows=False,
    ):
        """
        Args:
        """
        self.csv = pd.read_csv(csv)
        self.features_path = features_path
        self.features_path_audio = features_path_audio if features_path_audio != "" \
            else features_path
        self.caption = caption
        self.min_time = min_time
        self.feature_framerate = feature_framerate
        self.feature_framerate_3D = feature_framerate_3D
        self.we_dim = we_dim
        self.max_words = max_words
        self.min_words = min_words
        self.num_audio_frames = num_audio_frames
        self.we = we
        self.n_pair = n_pair
        self.fps = {'2d': feature_framerate, '3d': feature_framerate_3D}
        self.feature_path = {'2d': features_path}
        if features_path != '':
            self.feature_path['3d'] = features_path
        self.num_candidates = num_candidates
        self.random_audio_windows = random_audio_windows

    def __len__(self):
        return len(self.csv)

    def _zero_pad_tensor(self, tensor, size):
        if len(tensor) >= size:
            return tensor[:size]
        else:
            zero = np.zeros((size - len(tensor), self.we_dim), dtype=np.float32)
            return np.concatenate((tensor, zero), axis=0)

    def _zero_pad_audio(self, audio, max_frames):
        n_frames = audio.shape[1]
        if n_frames >= max_frames:
            return audio[:, 0:max_frames], int(max_frames)
        else:
            p = max_frames - n_frames
            audio_padded = np.pad(audio, ((0, 0), (0, p)), 'constant', constant_values=(0, 0))
            return audio_padded, n_frames

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
    """
    def _get_text(self, caption, n_pair_max):
        n_caption = len(caption['start'])
        k = n_pair_max
        starts = np.zeros(k)
        ends = np.zeros(k)
        text = th.zeros(k, self.max_words, self.we_dim)
        r_ind = np.random.choice(range(n_caption), k, replace=True)

        for i in range(k):
            ind = r_ind[i]
            text[i], starts[i], ends[i] = self._get_single_text(caption, ind)

        return text, starts, ends
    """
    def _get_single_text(self, caption, ind):
        start, end = ind, ind
        words = self._tokenize_text(caption['text'][ind])
        diff = caption['end'][end] - caption['start'][start]
        while len(words) < self.min_words or diff < self.min_time:
            if start > 0 and end < len(caption['end']) - 1:
                next_words = self._tokenize_text(caption['text'][end + 1])
                prev_words = self._tokenize_text(caption['text'][start - 1])
                d1 = caption['end'][end + 1] - caption['start'][start]
                d2 = caption['end'][end] - caption['start'][start - 1]
                if (self.min_time > 0 and d2 <= d1) or \
                    (self.min_time == 0 and len(next_words) <= len(prev_words)):
                    start -= 1
                    words.extend(prev_words)
                else:
                    end += 1
                    words.extend(next_words)
            elif start > 0:
                words.extend(self._tokenize_text(caption['text'][start - 1]))
                start -= 1
            elif end < len(caption['end']) - 1:
                words.extend(self._tokenize_text(caption['text'][end + 1]))
                end += 1
            else:
                break
            diff = caption['end'][end] - caption['start'][start]
        return self._words_to_we(words), \
            caption['start'][start], caption['end'][end]


    def _get_video(self, vid_path, s, e, video_id):
        feature_path = {}
        video = {}
        output = {}
        for k in self.feature_path:
            feature_path[k] = os.path.join(self.feature_path[k], vid_path, video_id + "_{}.npz".format(k))
            np_arr = np.load(feature_path[k])['features']
            video[k] = th.from_numpy(np_arr).float()
            output[k] = th.zeros(len(s), video[k].shape[-1])

            start = int(s * self.fps[k])
            end = int(e * self.fps[k]) + 1
            slice = video[k][start:end]
            if len(slice) < 1:
                #print("missing visual feats; video_id: {}, start: {}, end: {}".format(feature_path[k], start, end))
                missing=1
            else:
                output[k] = F.normalize(th.max(slice, dim=0)[0], dim=0)

        return th.cat([output[k] for k in output], dim=1)

    def _find_nearest_candidates(self, caption, ind):
        start, end = ind, ind
        diff = caption['end'][end] - caption['start'][start]
        n_candidate = 1
        while n_candidate < self.num_candidates:
            if start == 0:
                return 0
            elif end == len(caption) - 1:
                return start - (self.num_candidates - n_candidate)
            elif caption['end'][end] - caption['start'][start - 1] < caption['end'][end + 1] - caption['start'][start]:
                start -= 1
            else:
                end += 1
            n_candidate += 1
        return start

    def _get_text(self, cap):
        #cap = pd.read_csv(caption)
        ind = random.randint(0, len(cap) - 1)
        if self.num_candidates == 1:
            #words = self.words_to_ids(cap['text'].values[ind])
            words = self._tokenize_text(cap['text'][ind])
        else:
            #words = th.zeros(self.num_candidates, self.max_words, dtype=th.long)
            words = th.zeros(self.num_candidates, self.max_words, self.we_dim)
            cap_start = self._find_nearest_candidates(cap, ind)
            for i in range(self.num_candidates):
                candidate_w = cap['text'].values[max(0, min(len(cap['text']) - 1, cap_start + i))]
                word_token = self._tokenize_text(candidate_w)
                words[i] = self._words_to_we(word_token)#self.words_to_ids()
        start, end = cap['start'].values[ind], cap['end'].values[ind]
        # TODO: May need to be improved for edge cases.
        if end - start < self.min_time:
            diff = self.min_time - end + start
            start = max(0, start - diff / 2)
            end = start + self.min_time
        return words, int(start), int(end)

    def __getitem__(self, idx):
        vid_path = self.csv['path'].values[idx].replace("None/", "")
        video_id = vid_path.split("/")[-1]
        #audio_path = os.path.join(self.features_path_audio, vid_path, video_id + "_spec.npz")
        #mel_spec = np.load(audio_path)['arr_0']

        #video_path = os.path.join(self.video_root, video_file)
        text, start, end = self._get_text(self.caption[video_id])
        video = self._get_video(vid_path, start, end, video_id)
        #video = self._get_video(video_path, start, end)
        return {'video': video, 'text': text}
