from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn as nn
import torch as th
import torch.nn.functional as F
from model_davenet import load_DAVEnet


class Net(nn.Module):
    def __init__(
            self,
            embd_dim=4096,
            video_dim=4096,
            we_dim=300,
            tri_modal=False,
            tri_modal_fuse=False,
            cluster_size=256,
            layer=0,
            project=0,
            project_dim=6000,
            multi_cluster=0,
            recon=0,
            withMLP=0,
            recon_size=768,

    ):
        super(Net, self).__init__()
        self.DAVEnet = load_DAVEnet()
        self.DAVEnet_projection = nn.Linear(1024, embd_dim)
        self.GU_audio = Gated_Embedding_Unit(1024, 1024)
        self.GU_video = Gated_Embedding_Unit(video_dim, embd_dim)
        if tri_modal and not tri_modal_fuse:
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim)
            self.GU_text_captions = Gated_Embedding_Unit(embd_dim, embd_dim)

        elif tri_modal_fuse:
            self.DAVEnet_projection = nn.Linear(1024, embd_dim // 2)
            self.text_pooling_caption = Sentence_Maxpool(we_dim, embd_dim // 2)
            self.GU_audio_text = Fused_Gated_Unit(embd_dim // 2, embd_dim)
        self.tri_modal = tri_modal
        self.tri_modal_fuse = tri_modal_fuse
        self.project = project
        self.withMLP = withMLP
        self.recon_size = recon_size
        if withMLP==1:
            if project==0:
                self.classification = nn.Linear(embd_dim, project_dim, bias=False) #4096,256
                self.classification2 = nn.Linear(embd_dim, project_dim, bias=False)  # 4096,256
                self.classification3 = nn.Linear(embd_dim, project_dim, bias=False)  # 4096,256
            else:

                self.projection_head = nn.Sequential(
                    nn.Linear(embd_dim, embd_dim//8),
                    nn.BatchNorm1d(embd_dim//8),
                    nn.ReLU(inplace=True),
                    nn.Linear(embd_dim//8, cluster_size),
                )
                """
                self.projection_head2 = nn.Sequential(
                    nn.Linear(embd_dim, embd_dim),
                    nn.BatchNorm1d(embd_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embd_dim, cluster_size),
                )
                self.projection_head3 = nn.Sequential(
                    nn.Linear(embd_dim, embd_dim),
                    nn.BatchNorm1d(embd_dim),
                    nn.ReLU(inplace=True),
                    nn.Linear(embd_dim, cluster_size),
                )
                """
                self.classification = nn.Linear(cluster_size, project_dim, bias=False)
                #self.classification2 = nn.Linear(cluster_size, project_dim, bias=False)  # 4096,256
                #self.classification3 = nn.Linear(cluster_size, project_dim, bias=False)  # 4096,256

        self.layer=layer
        self.recon = recon
        if recon:
            inp_dim = embd_dim

            self.recon_v = nn.Sequential(
                nn.Linear(inp_dim, recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(recon_size, video_dim),
                nn.ReLU(inplace=True)
            )
            self.recon_a = nn.Sequential(
                nn.Linear(inp_dim, recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(recon_size, 1024),
                nn.ReLU(inplace=True)
            )
            self.recon_t = nn.Sequential(
                nn.Linear(inp_dim, recon_size),
                nn.ReLU(inplace=True),
                nn.Linear(recon_size, embd_dim),
                nn.ReLU(inplace=True)
            )
            self.mse = nn.MSELoss(reduction='none')


    def save_checkpoint(self, path):
        th.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        try:
            self.load_state_dict(th.load(path, map_location='cpu'))
        except Exception as e:
            print(e)
            print("IGNORING ERROR, LOADING MODEL USING STRICT=FALSE")
            self.load_state_dict(th.load(path, map_location='cpu'), strict=False)
        print("Loaded model checkpoint from {}".format(path))

    def forward(self, video, audio_input, nframes, text=None):
        video_gt = video
        video = self.GU_video(video)
        if self.recon:
            video_recon = self.recon_v(video)
        audio = self.DAVEnet(audio_input)
        if not self.training:  # controlled by net.train() / net.eval() (use for downstream tasks)
            # Mean-pool audio embeddings and disregard embeddings from input 0 padding
            pooling_ratio = round(audio_input.size(-1) / audio.size(-1))
            nframes.div_(pooling_ratio)
            audioPoolfunc = th.nn.AdaptiveAvgPool2d((1, 1)) #
            #audioPoolfunc = th.nn.AdaptiveMaxPool2d((1, 1))
            audio_outputs = audio.unsqueeze(2)
            pooled_audio_outputs_list = []
            for idx in range(audio.shape[0]):
                nF = max(1, nframes[idx])
                pooled_audio_outputs_list.append(audioPoolfunc(audio_outputs[idx][:, :, 0:nF]).unsqueeze(0))
            audio = th.cat(pooled_audio_outputs_list).squeeze(3).squeeze(2)
        else:
            audio = audio.mean(dim=2)  # this averages features from 0 padding too

        if self.tri_modal_fuse:
            text = self.text_pooling_caption(text)
            audio = self.DAVEnet_projection(audio)
            audio_text = self.GU_audio_text(audio, text)
            return audio_text, video

        # Gating in lower embedding dimension (1024 vs 4096) for stability with mixed-precision training
        audio_gt = audio
        audio = self.GU_audio(audio)
        audio = self.DAVEnet_projection(audio)
        if self.recon:
            audio_recon = self.recon_a(audio)
        if self.tri_modal and not self.tri_modal_fuse:
            text_gt = self.text_pooling_caption(text)
            text = self.GU_text_captions(text_gt)
            #fushed = (audio+text+video)/3
            # video_c2 = self.layer2(video)
            #"""
            if self.recon:
                text_recon = self.recon_t(text)


            if self.layer==1:
                video_c = self.layer1(video)
                audio_c = self.layer2(audio)
                text_c = self.layer3(text)
            else:
                if self.withMLP==1:
                    if self.project==1:
                        video_c = self.projection_head(video)
                        video_c = nn.functional.normalize(video_c, dim=1, p=2)
                    else:
                        video_c = nn.functional.normalize(video, dim=1, p=2)
                    video_c = self.classification(video_c)

                    #
                    if self.project == 1:
                        audio_c = self.projection_head(audio)
                        audio_c = nn.functional.normalize(audio_c, dim=1, p=2)
                    else:
                        audio_c = nn.functional.normalize(audio, dim=1, p=2)
                    audio_c = self.classification(audio_c)

                    #text_c = self.projection_head(text)
                    if self.project == 1:
                        text_c = self.projection_head(text)
                        text_c = nn.functional.normalize(text_c, dim=1, p=2)
                    else:
                        text_c = nn.functional.normalize(text, dim=1, p=2)
                    text_c = self.classification(text_c)
                #else:
                #    audio_c = video_c = text_c = audio
            #"""
            #fushed = (audio_c + text_c + video_c) / 3

            #fushed = self.projection_head(fushed)
            #fushed = nn.functional.normalize(fushed, dim=1, p=2)
            #video_c = audio_c = text_c= fushed#self.classification(fushed)
            if self.recon:
                mse_v = th.mean(self.mse(video_recon, video_gt), dim=-1)
                mse_a = th.mean(self.mse(audio_recon, audio_gt), dim=-1)
                mse_t = th.mean(self.mse(text_recon, text_gt), dim=-1)
                if self.withMLP == 1:
                    return audio, video, text, audio_c, video_c, text_c, mse_v + mse_a + mse_t
                else:
                    return audio, video, text, mse_v + mse_a + mse_t
            return audio, video, text, text#, audio_c, video_c, text_c
            #return audio, video, text 
        return audio, video


class Gated_Embedding_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Gated_Embedding_Unit, self).__init__()
        self.fc = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, x):
        x = self.fc(x)
        x = self.cg(x)
        return x


class Fused_Gated_Unit(nn.Module):
    def __init__(self, input_dimension, output_dimension):
        super(Fused_Gated_Unit, self).__init__()
        self.fc_audio = nn.Linear(input_dimension, output_dimension)
        self.fc_text = nn.Linear(input_dimension, output_dimension)
        self.cg = Context_Gating(output_dimension)

    def forward(self, audio, text):
        audio = self.fc_audio(audio)
        text = self.fc_text(text)
        x = audio + text
        x = self.cg(x)
        return x


class Context_Gating(nn.Module):
    def __init__(self, dimension):
        super(Context_Gating, self).__init__()
        self.fc = nn.Linear(dimension, dimension)

    def forward(self, x):
        x1 = self.fc(x)
        x = th.cat((x, x1), 1)
        return F.glu(x, 1)


class Sentence_Maxpool(nn.Module):
    def __init__(self, word_dimension, output_dim):
        super(Sentence_Maxpool, self).__init__()
        self.fc = nn.Linear(word_dimension, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return th.max(x, dim=1)[0]
