from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch.nn.functional as F
import torch as th
import numpy as np
eps = 1e-7

class MMS_loss(th.nn.Module):
    def __init__(self):
        super(MMS_loss, self).__init__()

    def forward(self, S,audio,video, margin=0.001):
        #print(audio.shape)
        #print(video.shape)

        #video = video.view(64,4, video.shape[-1])[:,0,:].squeeze()
        #audio = audio.view(64, 4, audio.shape[-1])[:, 0, :].squeeze()

        #print(video.shape)
        #video = video.permute(1, 0, 2) #4*64*4096
        #audio = audio.view(64,4, audio.shape[-1])
        video_embd = audio
        text_embd = video
        deltas = margin * th.eye(S.size(0)).cuda()#.to(S.device) #batch size eye
        #S = th.matmul(audio, video.t())  # 256*4096
        S = S - deltas #???
        """
        pseudo_v = pseudo_v.cpu().detach().numpy()
        soft = th.nn.Softmax(dim=1)(S)
        #pseudo_a = pseudo_v.cpu().detach().numpy()
        z_arr = np.ones((256, 256), dtype=float)
        for i in range(256):
            result, = np.where(pseudo_v == pseudo_v[i])
            # print(result)
            for r in result:
                # print(r)
                if i==r:
                    z_arr[i][r] = 1#-1000
                else:
                    #if S[i][r]>0:
                    z_arr[i][r] = 1#-soft[i][r]#0.001
                    #print(1-soft[i][r])
            # break
        #print(z_arr)
        z_arr = th.from_numpy(z_arr).type(th.FloatTensor).to(S.device)#z_arr.cuda()
        """
        """
        target = th.LongTensor(list(range(S.size(0)))).cuda()#.to(S.device) #0 to batch size list of numbers
        #print(target)
        #print(pseudo_a)
        #target_a = th.LongTensor(pseudo_a).to(S.device)
        #print(target_a)
        #target_v = th.LongTensor(pseudo_v).to(S.device)


        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), target) #softmax on feature
        C2I_loss = F.nll_loss(F.log_softmax(S.t(), dim=1), target)
        loss = I2C_loss + C2I_loss

        #I2C_loss = th.nn.BCELoss()(F.softmax(S, dim=1), z_arr)  # softmax on feature
        #C2I_loss = th.nn.BCELoss()(F.softmax(S.t(), dim=1), z_arr)
        #loss = I2C_loss + C2I_loss

        #return loss
        #"""
        #"""
        #video_embd = pseudo_v
        #text_embd = pseudo_a
        x = th.matmul(video_embd, text_embd.t())

        x = S
        x = x.view(video_embd.shape[0], video_embd.shape[0], -1)  # batch*batch*1

        #print(S)
        #x = x.view(S.shape[0], S.shape[0], -1)  # batch*batch*1
        nominator = x * th.eye(x.shape[0])[:, :, None].cuda()  # correct pairs, assume batches are same video
        #nominator = x * z_arr[:, :, None]
        #print(z_arr)
        #print(nominator)
        # replace eye by our one hot cluster label
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        #print(nominator)
        #p = x * z_arr[:, :, None]
        #pos = th.logsumexp(pos, dim=1)

        #pos = th.cat((p, p.permute(1, 0, 2)), dim=1).view(p.shape[0], -1)
        #pos = th.logsumexp(pos, dim=1)

        #x = x * z_arr[:, :, None]
        denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        #print(nominator)
        #print(denominator)
        return th.mean(denominator- nominator )
        #"""
        """
        numerator = th.logsumexp(th.diag(S).view(-1, 1), dim=1) # only  diagnal
        #print(th.diag(S).shape)
        #print(th.diag(S).view(-1, 1).shape) # 256*1
        #print(numerator.shape) #[256]
        denominator = th.logsumexp(th.cat([S, S.t()], dim=1), dim=1)
        #print(th.cat([S, S.t()], dim=1).shape)
        #print(denominator.shape) #256
        loss = th.mean(denominator - numerator)
        print(numerator)
        print(denominator)
        """
        #return loss


        #return loss



