from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import random
import os
import time
import pdb
import pickle
import numpy as np
from args import get_args
from functools import partial
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True)
from gensim.models.keyedvectors import KeyedVectors
import torch.nn as nn
import torch as th
th.backends.cudnn.benchmark = True
from torch.utils.data import DataLoader
import torch.optim as optim
from mycode.sink_pseu import create_pseu_single
from mycode.sink_pseu import create_pseu
from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from ucf_dataloader import UCF_DataLoader
from hmdb_dataloader import HMDB_DataLoader
from lsmdc_dataloader import LSMDC_DataLoader
from cvpr19_dataloader import CVPR19_DataLoader
from model_tri_c_clean_sp import Net
from loss import MMS_loss
from metrics import compute_metrics, print_computed_metrics, AverageMeter
from datetime import datetime
import math
from torch.optim.lr_scheduler import LambdaLR
from video_evaluation import evaluate_recall_youcook

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)



args = get_args()
if args.verbose:
    print(args)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# predefining random initial seeds
th.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

if args.checkpoint_dir != '' and not(os.path.isdir(args.checkpoint_dir)):
    os.mkdir(args.checkpoint_dir)

caption = None
if not(args.youcook) and not(args.msrvtt) and not(args.lsmdc):
    if not args.random_audio_windows:
        print('Loading HowTo100M captions: {}'.format(args.caption_path))
        caption = pickle.load(open(args.caption_path, 'rb'))
        print('done')

we = None
if args.tri_modal or not args.random_audio_windows:
    print('Loading word vectors: {}'.format(args.word2vec_path))
    we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
    print('done')

if args.youcook:
    dataset = Youcook_DataLoader(
        data=args.youcook_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.youcook_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
elif args.msrvtt:
    dataset = MSRVTT_DataLoader(
        data_path=args.msrvtt_train_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=True,
        tri_modal=args.tri_modal,
    )
elif args.lsmdc:
    dataset = LSMDC_DataLoader(
        data_path=args.lsmdc_train_path,
        we=we,
        max_words=args.max_words,
        num_frames_multiplier=args.lsmdc_num_frames_multiplier,
        we_dim=args.we_dim,
        tri_modal=args.tri_modal,
    )
else:
    if args.cooperative==0:
        dataset = Youtube_DataLoader(
            csv=args.train_csv,
            features_path=args.features_path,
            features_path_audio=args.features_path_audio,
            caption=caption,
            min_time=args.min_time,
            max_words=args.max_words,
            min_words=args.min_words,
            feature_framerate=args.feature_framerate,
            we=we,
            we_dim=args.we_dim,
            n_pair=args.n_pair,
            num_audio_frames=args.howto_audio_frames,
            random_audio_windows=args.random_audio_windows,
            finetune_video=args.finetune_video,
            video_interp=args.video_interp
        )
if args.cooperative==0:
    dataset_size = len(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_thread_reader,
        shuffle=True,
        batch_sampler=None,
        drop_last=True,
    )
if args.eval_youcook:
    dataset_val = Youcook_DataLoader(
        data=args.youcook_val_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.youcook_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_lsmdc:
    dataset_lsmdc = LSMDC_DataLoader(
        data_path=args.lsmdc_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.lsmdc_num_frames_multiplier,
        tri_modal=args.tri_modal,
    )
    dataloader_lsmdc = DataLoader(
        dataset_lsmdc,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
    )
if args.eval_msrvtt:
    msrvtt_testset = MSRVTT_DataLoader(
        data_path=args.msrvtt_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=False,
        tri_modal=args.tri_modal,
        finetune_video=args.finetune_video,
        video_interp=args.video_interp
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
if args.eval_ucf:
    ucf_testset = UCF_DataLoader(
        data_path=args.ucf_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=False,
        tri_modal=args.tri_modal,
        finetune_video=args.finetune_video,
        video_interp=args.video_interp
    )
    dataloader_ucf = DataLoader(
        ucf_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
if args.eval_hmdb:
    hmdb_testset = HMDB_DataLoader(
        data_path=args.hmdb_test_path,
        we=we,
        max_words=args.max_words,
        we_dim=args.we_dim,
        num_frames_multiplier=args.msrvtt_num_frames_multiplier,
        training=False,
        tri_modal=args.tri_modal,
        finetune_video=args.finetune_video,
        video_interp=args.video_interp
    )
    dataloader_hmdb = DataLoader(
        hmdb_testset,
        batch_size=1000,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
if args.eval_cross:
    step_path = 'data/step.json'
    cross_testset = CVPR19_DataLoader(
        # csv='vids_27.csv',
        csv='/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/cvpr19_test.csv',
        features_path='/nobackup/users/brian27/howto100m/vids_feature',
        # features_path = '/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/features_2d',
        # features_path_3D = '/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/features_3d',
        annot_path='/nobackup/users/brian27/howto100m/anno',  # '/nobackup/users/brian27/CrossTask/crosstask_release/Y-1',
        steps_path=step_path,
        audio_path='/nobackup/users/brian27/howto100m/audio_feature',  # '/home/brian27/nobackup/CrossTask/audio_feature_new',#
        annot_path_time='/nobackup/users/brian27/CrossTask/crosstask_release/annotations',
        cook_path='/nobackup/users/brian27/CrossTask/crosstask_release/cook.txt',
        with_audio=1,
        we=we
        # features_path_3D='howto100m_crosstask_eval/features_3d'
    )
    dataloader_cross = DataLoader(
        cross_testset,
        batch_size=1,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
net = Net(
    embd_dim=args.embd_dim,
    video_dim=args.feature_dim,
    we_dim=args.we_dim,
    tri_modal=args.tri_modal,
    tri_modal_fuse=args.tri_modal_fuse,
    cluster_size=args.cluster_size,
    layer=args.layer,
    project=args.project,
    project_dim=args.project_dim,
    multi_cluster=args.multi_cluster,
    recon=args.recon,
    recon_b=args.recon_b,
    finetune_video=args.finetune_video,
    multi_head=args.multi_head,
    joint_cluster=args.joint_cluster,
    output_norm=args.output_norm,
    recon_cross=args.recon_cross
)
if args.cluster_a:
    assert args.project_dim == args.embd_dim

# Optimizers + Loss
if args.loss == 0:
    loss_op = MMS_loss()
net.cuda()
loss_op.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)

scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, len(dataloader) * args.epochs)


if args.apex_level == 0:
    apex = False
elif args.apex_level == 1:
    from apex import amp, optimizers
    net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    apex = True
net = th.nn.DataParallel(net)
net.train()

if args.pretrain_path != '' and args.apex_level == 1:
    amp_checkpoint_path = os.path.join(os.path.dirname(args.pretrain_path), 'amp_checkpoint.pt')
    checkpoint = th.load(amp_checkpoint_path, map_location='cpu')
    net.module.load_state_dict(checkpoint['net'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint["scheduler"])
    amp.load_state_dict(checkpoint['amp'])
    print("Loaded AMP checkpoint")
elif args.pretrain_path != '' and args.apex_level == 0:
    net.module.load_checkpoint(args.pretrain_path)

if args.verbose:
    print('Starting training loop ...')


def TrainOneBatch(model, opt, data, loss_fun,queue_v,queue_a,queue_t,use_the_queue, scheduler, epoch, apex=False):
    with th.no_grad():
        w = model.module.classification.weight.data.clone()  # 3000*128
        w = nn.functional.normalize(w, dim=1, p=2)
        model.module.classification.weight.copy_(w)
        """
        w = model.module.classification2.weight.data.clone()  # 3000*128
        w = nn.functional.normalize(w, dim=1, p=2)
        model.module.classification2.weight.copy_(w)
        w = model.module.classification3.weight.data.clone()  # 3000*128
        w = nn.functional.normalize(w, dim=1, p=2)
        model.module.classification3.weight.copy_(w)
        """
    #cluster = args.cluster

    video = data['video'].cuda()
    audio = data['audio'].cuda()
    nframes = data['nframes'].cuda()

    if args.finetune_video:
        video = video.view(-1, video.shape[-2], video.shape[-1])
    else:
        video = video.view(-1, video.shape[-1])
    audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
    nframes = nframes.view(-1)
    opt.zero_grad()
    bs = video.size(0)  # 256
    with th.set_grad_enabled(True):
        if args.tri_modal:
            text = data['text'].cuda()
            text = text.view(-1, text.shape[-2], text.shape[-1])
            if args.tri_modal_fuse: # AVLnet-Text audio-text fusion model
                audio_text, video = model(video, audio, nframes, text)
                sim_audiotext_video = th.matmul(audio_text, video.t())
                loss = loss_fun(sim_audiotext_video)
            else: # AVLnet-Text independent audio and text branches
                if args.recon:
                    audio, video, text, out_a, out_v, out_t, recon_loss = model(video, audio, nframes, text)
                    if args.recon_b:
                        recon_w = 20
                    else:
                        recon_w = 50
                    recon_loss = th.mean(recon_loss)*recon_w
                else:
                    audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                if args.joint_cluster:
                    out_a = out_v = out_t = (out_a + out_t + out_v) / 3
                video_out = out_v.detach()
                audio_out = out_a.detach()
                text_out = out_t.detach()
                #pdb.set_trace()
                if args.use_queue == 1:

                    if queue_v is not None: #no queue in first round
                        if use_the_queue or not th.all(queue_v[ -1, :] == 0): # queue[2,3840,128] if use the queue or the queue is full
                            use_the_queue = True
                            video_out = th.cat((th.mm(queue_v,model.module.classification.weight.t()), video_out)) #queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000
                            audio_out = th.cat((th.mm(queue_v, model.module.classification.weight.t()), audio_out))
                            text_out = th.cat((th.mm(queue_v, model.module.classification.weight.t()), text_out))
                        # fill the queue
                        queue_v[ bs:] = queue_v[ :-bs].clone() # move 0-6 to 1-7 place
                        #print('video_shape: ', video.shape)
                        #print('queue_v_shape: ', queue_v[:bs].shape)
                        #print('batch_size: ', bs)
                        queue_v[ :bs] = video.detach() #12288,4096 in howto
                        queue_a[bs:] = queue_a[:-bs].clone()  # move 0-6 to 1-7 place
                        queue_a[:bs] = audio.detach()
                        queue_t[bs:] = queue_t[:-bs].clone()  # move 0-6 to 1-7 place
                        queue_t[:bs] = text.detach()



                if args.fuse_videoaudio_additive: # only used for fine-tuning
                    audio_video = audio + video
                    sim_text_audiovideo = th.matmul(text, audio_video.t())
                    loss = loss_fun(sim_text_audiovideo)
                else: #here, was modified, need to change it back
                    if args.soft_contrast_only:
                        sim_audio_video = th.matmul(audio, video.t())
                        sim_audio_text = th.matmul(audio, text.t())
                        sim_text_video = th.matmul(text, video.t())
                    elif args.proto_nce == 1:
                        sim_audio_video = th.matmul(out_a, out_v.t())
                        sim_audio_text = th.matmul(out_a, out_t.t())
                        sim_text_video = th.matmul(out_t, out_v.t())

                        loss = loss_fun(sim_text_video) + loss_fun(sim_audio_text) + loss_fun(sim_audio_video)
                    else:
                        sim_audio_video = th.matmul(audio, video.t())
                        sim_audio_text = th.matmul(audio, text.t())
                        sim_text_video = th.matmul(text, video.t())
                        if args.no_audio:
                            loss = loss_fun(sim_text_video)
                        else:
                            loss = loss_fun(sim_text_video) + loss_fun(sim_audio_text) + loss_fun(sim_audio_video)
                    #"""
                    if args.cluster == 1 and epoch>=args.start_cluster:
                        #print('start_cluster')
                        #true batch 4096 different images, 3 images in one batch
                        #if args.multi_cluster==0:
                        pseudo_v, pseudo_a, pseudo_t, Q_v, Q_a, Q_t  = create_pseu_single(video_out, audio_out, text_out, args.cluster_size)
                        Q_a = Q_t = Q_v


                        #else:
                        #pseudo_v, pseudo_a, pseudo_t, Q_v, Q_a, Q_t = create_pseu(video_out, audio_out,text_out,args.cluster_size)
                        if args.soft_label==0:
                            loss_val = th.nn.CrossEntropyLoss()(out_a, pseudo_v) + \
                                       th.nn.CrossEntropyLoss()(out_v, pseudo_v) + \
                                       th.nn.CrossEntropyLoss()(out_t, pseudo_v)
                            loss_val = loss_val / 3

                        else:
                            subloss=0
                            softmax = nn.Softmax(dim=1).cuda()
                            if args.cluster_a:
                                p_v = softmax(video * 10)
                            else:
                                p_v = softmax(out_v*10)  # v=1, 32*3000

                            #pdb.set_trace()
                            if args.no_audio==0:
                                subloss -= th.mean(th.sum(Q_a * th.log(p_v), dim=1))
                            #pdb.set_trace()
                            subloss -= th.mean(th.sum(Q_t * th.log(p_v), dim=1))

                            #pdb.set_trace()
                            if args.cluster_a:
                                p_a = softmax(audio * 10)
                            else:
                                p_a = softmax(out_a*10)  # v=1, 32*3000
                            if args.no_audio == 0:
                                subloss -= th.mean(th.sum(Q_v * th.log(p_a), dim=1))
                                subloss -= th.mean(th.sum(Q_t * th.log(p_a), dim=1))

                            if args.cluster_a:
                                p_t = softmax(text * 10)
                            else:
                                p_t = softmax(out_t*10)  # v=1, 32*3000
                            if args.no_audio == 0:
                                subloss -= th.mean(th.sum(Q_a * th.log(p_t), dim=1))
                            subloss -= th.mean(th.sum(Q_v * th.log(p_t), dim=1))
                            if args.self_prediction==1:
                                subloss -= th.mean(th.sum(Q_v * th.log(p_v), dim=1))
                                subloss -= th.mean(th.sum(Q_a * th.log(p_a), dim=1))
                                subloss -= th.mean(th.sum(Q_t * th.log(p_t), dim=1))
                                #print('p_v', p_v)
                                #print('p_a', p_a)
                                #print('p_t', p_t)
                            if args.pseudo_contrast==1:
                                sim_audio_video_P = th.matmul(Q_a, Q_v.t())
                                sim_audio_text_P = th.matmul(Q_a, Q_t.t())
                                sim_text_video_P = th.matmul(Q_t, Q_v.t())

                                loss += loss_fun(sim_text_video_P) + loss_fun(sim_audio_text_P) + loss_fun(sim_audio_video_P)
                            if args.soft_contrast==1:
                                loss_soft_contrast=0
                                sim_audio_video_Q = th.matmul(Q_a, Q_v.t())
                                sim_audio_text_Q = th.matmul(Q_a, Q_t.t())
                                sim_text_video_Q = th.matmul(Q_t, Q_v.t())
                                #print('sim_audio_video_Q',sim_audio_video_Q)
                                #print('sim_audio_text_Q',sim_audio_text_Q)
                                #print('sim_text_video_Q',sim_text_video_Q)
                                deltas = 0.001 * th.eye(Q_v.size(0)).cuda()  # .to(S.device) #batch size eye


                                x = sim_audio_video_Q#sim_audio_video -deltas
                                x = x.view(video.shape[0], video.shape[0], -1)
                                if args.nce==0:
                                    sim_audio_video_Q = sim_audio_video_Q.view(video.shape[0], video.shape[0], -1)
                                    nominator = x * sim_audio_video_Q.detach()  # correct pairs, assume batches are same video
                                else:
                                    nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
                                nominator = nominator.sum(dim=1)
                                nominator = th.logsumexp(nominator, dim=1)
                                denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
                                denominator = th.logsumexp(denominator, dim=1)
                                loss_soft_contrast+=th.mean(denominator - nominator)
                                x = sim_audio_text_Q#sim_audio_text - deltas
                                x = x.view(video.shape[0], video.shape[0], -1)
                                if args.nce == 0:
                                    sim_audio_text_Q = sim_audio_text_Q.view(video.shape[0], video.shape[0], -1)
                                    nominator = x * sim_audio_text_Q.detach()  # correct pairs, assume batches are same video
                                else:
                                    nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
                                nominator = nominator.sum(dim=1)
                                nominator = th.logsumexp(nominator, dim=1)
                                denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
                                denominator = th.logsumexp(denominator, dim=1)
                                loss_soft_contrast += th.mean(denominator - nominator)
                                x = sim_text_video_Q#sim_text_video  - deltas
                                x = x.view(video.shape[0], video.shape[0], -1)
                                if args.nce == 0:
                                    sim_text_video_Q = sim_text_video_Q.view(video.shape[0], video.shape[0], -1)
                                    nominator = x * sim_text_video_Q.detach()  # correct pairs, assume batches are same video
                                else:
                                    nominator = x * th.eye(x.shape[0])[:, :, None].cuda()
                                nominator = nominator.sum(dim=1)
                                nominator = th.logsumexp(nominator, dim=1)
                                denominator = th.cat((x, x.permute(1, 0, 2)), dim=1).view(x.shape[0], -1)
                                denominator = th.logsumexp(denominator, dim=1)
                                loss_soft_contrast += th.mean(denominator - nominator)
                                loss_soft_contrast/=3
                                if args.soft_contrast_only:
                                    loss=loss_soft_contrast
                                else:
                                    loss += loss_soft_contrast
                            if args.nce_only==1:
                                loss_val=0
                            else:
                                loss_val = subloss/3
                        print('loss: ', loss)
                        print('loss_val: ', loss_val)

                        if args.switch_loss_h==1:
                            if epoch%2==0:
                                loss_val=0
                            else:
                                loss=0
                        if args.switch_loss_s==1:
                            if epoch%2==0:
                                loss_val*=0.3
                                loss*=0.7
                            else:
                                loss_val *= 0.7
                                loss *= 0.3
                        loss+= loss_val
                        if args.pure_cluster==1:
                            loss = loss_val
                        if args.recon:
                            print('recon_loss: ', recon_loss)
                            loss += recon_loss
                    #"""


        else:
            audio, video = model(video, audio, nframes)
            sim_matrix = th.matmul(audio, video.t())
            loss = loss_fun(sim_matrix)
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    # cancel some gradients
    freeze_prototypes_niters = 313
    freeze_prototypes_epoch = 1
    #print('cur epoch',epoch)
    #if iteration < freeze_prototypes_niters:
    if epoch <= freeze_prototypes_epoch:
        for name, p in model.named_parameters():
            if "classification" in name:
                p.grad = None

    opt.step()
    scheduler.step()
    return loss.item(),queue_v,queue_a,queue_t, use_the_queue


def Eval_retrieval(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        for data in eval_dataloader:
            video = data['video'].cuda()
            audio = data['audio'].cuda()
            nframes = data['nframes'].cuda()
            if args.tri_modal:
                text = data['text'].cuda()
                if args.tri_modal_fuse==1: # AVLnet-Text
                    audio_text, video = model(video, audio, nframes, text)
                    m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                else: #args.fuse_videoaudio_additive: # eval T->V+A for AVLnet-Text indep. model
                    if args.output_norm:
                        audio, video, text, out_a, out_v, out_t, out_a2, out_v2, out_t2 = model(video, audio, nframes, text)
                        if args.eval_msrvtt == 1:
                            audio_video = video  # audio
                        else:
                            audio_video = video + audio
                        m = th.matmul(text, audio_video.t()).cpu().detach().numpy()

                        store_data = {'audio': audio.cpu().detach().numpy(),
                                      'video': video.cpu().detach().numpy(),
                                      'text': text.cpu().detach().numpy(),
                                      'out_a': out_a.cpu().detach().numpy(),
                                      'out_v': out_v.cpu().detach().numpy(),
                                      'out_t': out_t.cpu().detach().numpy(),
                                      'out_a2': out_a2.cpu().detach().numpy(),
                                      'out_v2': out_v2.cpu().detach().numpy(),
                                      'out_t2': out_t2.cpu().detach().numpy(),
                                      'video_id': data['video_id']
                                      }
                        pickle.dump(store_data, open('./temp_data/%s.pkl' % dataset_name, 'wb'))
                        print('Stored Data')
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                        if args.eval_msrvtt == 1:
                            audio_video = video  # audio
                        else:
                            audio_video = video + audio
                        m = th.matmul(text, audio_video.t()).cpu().detach().numpy()

                        store_data = {'audio': audio.cpu().detach().numpy(),
                                      'video': video.cpu().detach().numpy(),
                                      'text': text.cpu().detach().numpy(),
                                      'out_a': out_a.cpu().detach().numpy(),
                                      'out_v': out_v.cpu().detach().numpy(),
                                      'out_t': out_t.cpu().detach().numpy(),
                                      'video_id': data['video_id']
                                      }
                        pickle.dump(store_data, open('./temp_data/%s.pkl' % dataset_name, 'wb'))
                        print('Stored Data')
            else:
                audio, video = model(video, audio, nframes)
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
            metrics = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
            print_computed_metrics(metrics)

            if args.eval_youcook:
                evaluate_recall_youcook(text.cpu().detach().numpy(), audio_video.cpu().detach().numpy(), data['video_id'])


def Eval_retrieval_cross(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating retrieval on {} data'.format(dataset_name))
    with th.no_grad():
        store_data = {}  # 'video_id': [video_feats, text_feats, task, audio_feats]

        for i, data in enumerate(eval_dataloader):
            video = data['video'].cuda()[0]
            audio = th.zeros((4, 40, 1024), dtype=video.dtype)  # data['audio'].cuda()[0]
            text = data['steps'].cuda()[0]

            nframes = data['nframes'].cuda()[0]
            video_id = data['video_id'][0]
            task = data['task']
            print(i, video_id, video.shape, audio.shape, text.shape)

            audio, video, text, out_a, out_v, out_t, out_a2, out_v2, out_t2 = model(video, audio, nframes, text)

            store_data[video_id] = {#'audio': audio.cpu().detach().numpy(),
                          'video': video.cpu().detach().numpy(),
                          'text': text.cpu().detach().numpy(),
                          'task': task[0] }


        pickle.dump(store_data, open('./temp_data/%s.pkl' % dataset_name, 'wb'))
        print('Stored Data')



def Eval_classification(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating ZSL classification on {} data'.format(dataset_name))
    with th.no_grad():
        video_feats, text_feats, audio_feats, gt_classes, has_audio = [], [], [], [], []
        for data in eval_dataloader:
            gt_classes.append(data['class_id'])
            video = data['video'].cuda()
            audio = data['audio'].cuda()
            nframes = data['nframes'].cuda()
            has_audio.append(data['has_audio'].cuda())
            if args.tri_modal:
                text = eval_dataloader.dataset.class_embeds.cuda() #data['text'].cuda()
                if args.tri_modal_fuse==1: # AVLnet-Text
                    audio_text, video = model(video, audio, nframes, text)
                    video_feats.append(video)
                    text_feats.append(audio_text)
                    audio_feats.append(audio_text)
                    #m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                else: #args.fuse_videoaudio_additive: # eval T->V+A for AVLnet-Text indep. model
                    if args.output_norm:
                        audio, video, text, out_a, out_v, out_t, out_a2, out_v2, out_t2 = model(video, audio, nframes, text)
                        audio_video = video
                        video_feats.append(video)
                        text_feats.append(text)
                        audio_feats.append(audio)
                        #m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                        audio_video = video
                        video_feats.append(video)
                        text_feats.append(text)
                        audio_feats.append(audio)
                        #m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            else:
                audio, video = model(video, audio, nframes)
                video_feats.append(video)
                text_feats.append(audio)
                #m = th.matmul(audio, video.t()).cpu().detach().numpy()

        video_feats = th.cat(video_feats, 0)
        audio_feats = th.cat(audio_feats, 0)
        #text_feats = th.cat(text_feats, 0)
        text_feats = text_feats[0]
        has_audio = th.cat(has_audio, 0)

        m = th.matmul(text_feats, (video_feats).t()).cpu().detach().numpy() #(T, V)
        #print(m.shape)
        argmaxes = np.argmax(m, 0)
        #print(argmaxes.shape)
        gt_classes = th.cat(gt_classes, 0).cpu().detach().numpy()[:, 0]
        has_audio = has_audio.cpu().detach().numpy()

        store_data = {'video': video_feats.cpu().detach().numpy(),
                      'text': text_feats.cpu().detach().numpy(),
                      'audio': audio_feats.cpu().detach().numpy(),
                      'gt_classes': gt_classes,
                      'has_audio': has_audio
                      }
        pickle.dump(store_data, open('./temp_data/%s.pkl' % dataset_name, 'wb'))
        print('Stored Data')

        print('Accuracy on {}:'.format(dataset_name), np.mean(argmaxes==gt_classes))

        y = gt_classes
        y_pred = m.T.argsort(1)

        accuracy_top5 = np.mean([l in p for l, p in zip(y, y_pred[:, -5:])])
        print('Top-5 Accuracy on {}:'.format(dataset_name), accuracy_top5)


def Eval_classification_nn(model, eval_dataloader, dataset_name):
    model.eval()
    print('Evaluating Nearest Neighbor classification on {} data'.format(dataset_name))
    with th.no_grad():
        video_feats, gt_classes, training, inp_feats = [], [], [], []
        for data in eval_dataloader:
            gt_classes.append(data['class_id'])
            training.append(data['training'])
            inp_feats.append(data['video'])
            video = data['video'].cuda()
            audio = data['audio'].cuda()
            nframes = data['nframes'].cuda()

            if args.tri_modal:
                text = eval_dataloader.dataset.class_embeds.cuda() #data['text'].cuda()
                if args.tri_modal_fuse==1: # AVLnet-Text
                    audio_text, video = model(video, audio, nframes, text)
                    video_feats.append(video)
                    #m = th.matmul(audio_text, video.t()).cpu().detach().numpy()
                else: #args.fuse_videoaudio_additive: # eval T->V+A for AVLnet-Text indep. model
                    if args.output_norm:
                        audio, video, text, out_a, out_v, out_t, out_a2, out_v2, out_t2 = model(video, audio, nframes, text)
                        audio_video = video
                        video_feats.append(video)
                        #m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                        audio_video = video
                        video_feats.append(video)
                        #m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
            else:
                audio, video = model(video, audio, nframes)
                video_feats.append(video)

        video_feats = th.cat(video_feats, 0)
        inp_feats = th.cat(inp_feats, 0)

        gt_classes = th.cat(gt_classes, 0).cpu().detach().numpy()[:, 0]
        training = th.cat(training, 0).cpu().detach().numpy()

        store_data = {'video': video_feats.cpu().detach().numpy(),
                      'gt_classes': gt_classes,
                      'training': training,
                      'inp_feats': inp_feats.cpu().detach().numpy(),
                      }
        pickle.dump(store_data, open('./temp_data/%s_nn.pkl' % dataset_name, 'wb'))
        print('Stored Data')

        video = store_data['video']
        training = store_data['training'][:, 0]
        gt_classes = store_data['gt_classes']

        training_vids = video[training == 1]  # (N1, F)
        training_classes = gt_classes[training == 1]

        testing_vids = video[training == 0]  # (N2, F)
        testing_classes = gt_classes[training == 0]

        m = np.matmul(testing_vids, training_vids.T)  # (N2, N1)

        argsorted = np.argsort(m, 1)  # (N2, N1)

        class_predictions = training_classes[argsorted]

        top_1 = class_predictions[:, -1]

        recall = np.mean(top_1 == testing_classes)
        print('NN Retreival R@1:', recall)

        recall_top5 = np.mean([l in p for l, p in zip(testing_classes, class_predictions[:, -5:])])
        print('NN Retreival R@5:', recall_top5)

        recall_top10 = np.mean([l in p for l, p in zip(testing_classes, class_predictions[:, -10:])])
        print('NN Retreival R@5:', recall_top10)




batch_time = AverageMeter()
data_time = AverageMeter()
queue_v = None
queue_a = None
queue_t = None

n_pair=1
for epoch in range(args.epochs):
    save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
        else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
    n_pair = 2**(save_epoch-1)
    if n_pair>=32:
        n_pair=32
    batch_size=int(4096/n_pair)
    if args.cooperative == 1:
        dataset = Youtube_DataLoader(
            csv=args.train_csv,
            features_path=args.features_path,
            features_path_audio=args.features_path_audio,
            caption=caption,
            min_time=args.min_time,
            max_words=args.max_words,
            min_words=args.min_words,
            feature_framerate=args.feature_framerate,
            we=we,
            we_dim=args.we_dim,
            n_pair=n_pair,
            num_audio_frames=args.howto_audio_frames,
            random_audio_windows=args.random_audio_windows,
        )
        dataset_size = len(dataset)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            batch_sampler=None,
            drop_last=True,
        )


    running_loss = 0.0
    if args.eval_youcook:
        Eval_retrieval(net, dataloader_val, 'YouCook2')
    if args.eval_msrvtt:
        Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
    if args.eval_lsmdc:
        Eval_retrieval(net, dataloader_lsmdc, 'LSMDC')
    if args.verbose:
        print('Epoch: %d' % epoch)
    end_time = time.time()

    #args. =3
    queue_l = args.queue_size
    if args.use_queue==1 and epoch >= args.start_queue and queue_v is None:  # will start at epoch 15
        queue_v = th.zeros(
            queue_l,  # 3840 //2 = 1920
            4096,  # 128
        ).cuda()
        queue_a = th.zeros(
            queue_l,  # 3840 //2 = 1920
            4096,  # 128
        ).cuda()
        queue_t = th.zeros(
            queue_l,  # 3840 //2 = 1920
            4096,  # 128
        ).cuda()
    save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
        else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
    use_the_queue = False
    for i_batch, sample_batch in enumerate(tqdm(dataloader)):
        data_load_time = time.time() - end_time
        data_time.update(data_load_time)

        iteration = epoch * len(dataloader) + i_batch  # 0

        batch_loss,queue_v,queue_a,queue_t,use_the_queue = TrainOneBatch(net, optimizer, sample_batch, loss_op,queue_v,queue_a,queue_t,use_the_queue,scheduler, save_epoch, apex)
        process_batch_time = time.time() - end_time
        batch_time.update(process_batch_time)
        running_loss += batch_loss
        if (i_batch + 1) % args.n_display == 0 and args.verbose:
            print('Epoch %d, Epoch status: %.4f, Training loss: %.4f' %
            (epoch + 1, args.batch_size * float(i_batch) / dataset_size,
            running_loss / args.n_display))
            print('Batch load time avg: %.4f, Batch process time avg: %.4f' %
            (data_time.avg, batch_time.avg))
            running_loss = 0.0
            # reset the load meters
            batch_time = AverageMeter()
            data_time = AverageMeter()
        end_time = time.time()
    save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
                 else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
    #for param_group in optimizer.param_groups:
    #    param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '':
        path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(save_epoch))
        net.module.save_checkpoint(path)
        if args.apex_level == 1:
            amp_checkpoint = {'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                              "scheduler": scheduler.state_dict(),
                            'amp': amp.state_dict()}
            th.save(amp_checkpoint, os.path.join(args.checkpoint_dir, 'amp_checkpoint.pt'))


if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2')
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
if args.eval_lsmdc:
    Eval_retrieval(net, dataloader_lsmdc, 'LSMDC')
if args.eval_ucf:
    Eval_classification(net, dataloader_ucf, 'UCF-101')
    #Eval_classification_nn(net, dataloader_ucf, 'UCF-101')
if args.eval_hmdb:
    Eval_classification(net, dataloader_hmdb, 'HMDB')
    #Eval_classification_nn(net, dataloader_hmdb, 'HMDB')
if args.eval_cross:
    Eval_retrieval_cross(net, dataloader_cross, 'Cross')
    #Eval_classification_nn(net, dataloader_hmdb, 'HMDB')