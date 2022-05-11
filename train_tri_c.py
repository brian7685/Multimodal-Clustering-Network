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
from lsmdc_dataloader import LSMDC_DataLoader
from model_tri_c import Net
from loss import MMS_loss
from metrics import compute_metrics, print_computed_metrics, AverageMeter
from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)

args = get_args()
if args.verbose:
    print(args)

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
        finetune_video=args.finetune_video
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
        finetune_video=args.finetune_video
    )
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        batch_size=1000,
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
    finetune_video=args.finetune_video
)

# Optimizers + Loss
if args.loss == 0:
    loss_op = MMS_loss()
net.cuda()
loss_op.cuda()
optimizer = optim.Adam(net.parameters(), lr=args.lr)
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
    amp.load_state_dict(checkpoint['amp'])
    print("Loaded AMP checkpoint")
elif args.pretrain_path != '' and args.apex_level == 0:
    net.module.load_checkpoint(args.pretrain_path)

if args.verbose:
    print('Starting training loop ...')

def TrainOneBatch(model, opt, data, loss_fun,queue_v,queue_a,queue_t,use_the_queue, iteration, epoch, apex=False):
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
                audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
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
                        if args.no_audio==1:
                            loss = loss_fun(sim_text_video)
                        if args.no_va==1:
                            loss = loss_fun(sim_text_video) + loss_fun(sim_audio_text)
                        else:
                            loss = loss_fun(sim_text_video) + loss_fun(sim_audio_text) + loss_fun(sim_audio_video)
                    #"""
                    if args.cluster == 1 and epoch>=args.start_cluster:
                        #print('start_cluster')
                        #true batch 4096 different images, 3 images in one batch
                        #if args.multi_cluster==0:
                        pseudo_v, pseudo_a, pseudo_t, Q_v, Q_a, Q_t  = create_pseu_single(video_out, audio_out, text_out, args.cluster_size)
                        #else:
                        #pseudo_v, pseudo_a, pseudo_t, Q_v, Q_a, Q_t = create_pseu(video_out, audio_out,text_out,args.cluster_size)
                        if args.soft_label==0:
                            loss_val = th.nn.CrossEntropyLoss()(out_a, pseudo_t[-bs:]) + \
                                       th.nn.CrossEntropyLoss()(out_v, pseudo_t[-bs:]) + \
                                       th.nn.CrossEntropyLoss()(out_t, pseudo_v[-bs:]) + \
                                       th.nn.CrossEntropyLoss()(out_t, pseudo_a[-bs:]) + \
                                       th.nn.CrossEntropyLoss()(out_a, pseudo_v[-bs:]) + \
                                       th.nn.CrossEntropyLoss()(out_v, pseudo_a[-bs:])
                        else:
                            subloss=0
                            softmax = nn.Softmax(dim=1).cuda()
                            p_v = softmax(out_v*10)  # v=1, 32*3000

                            #pdb.set_trace()
                            if args.no_audio==0:
                                subloss -= th.mean(th.sum(Q_a * th.log(p_v), dim=1))
                            #pdb.set_trace()
                            subloss -= th.mean(th.sum(Q_t * th.log(p_v), dim=1))

                            #pdb.set_trace()
                            p_a = softmax(out_a*10)  # v=1, 32*3000
                            if args.no_audio == 0:
                                subloss -= th.mean(th.sum(Q_v * th.log(p_a), dim=1))
                                subloss -= th.mean(th.sum(Q_t * th.log(p_a), dim=1))
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
    print('cur epoch',epoch)
    #if iteration < freeze_prototypes_niters:
    if epoch <= freeze_prototypes_epoch:
        for name, p in model.named_parameters():
            if "classification" in name:
                p.grad = None

    opt.step()
    #scheduler.step()
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
                    audio, video, text, out_a, out_v, out_t, out_a2, out_v2, out_t2 = model(video, audio, nframes, text)
                    if args.eval_msrvtt==1:
                        audio_video=video#audio
                    else:
                        audio_video = video+audio

                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()


                    store_data = {'audio': audio.cpu().detach().numpy(),
                                  'video': video.cpu().detach().numpy(),
                                  'text': text.cpu().detach().numpy(),
                                  'out_a': out_a.cpu().detach().numpy(),
                                  'out_v': out_v.cpu().detach().numpy(),
                                  'out_t': out_t.cpu().detach().numpy(),
                                  'out_a2': out_a2.cpu().detach().numpy(),
                                  'out_v2': out_v2.cpu().detach().numpy(),
                                  'out_t2': out_t2.cpu().detach().numpy()
                                  }
                    pickle.dump(store_data, open('./temp_data/%s.pkl' % dataset_name, 'wb'))
                    print('Stored Data')
            else:
                audio, video = model(video, audio, nframes)
                m = th.matmul(audio, video.t()).cpu().detach().numpy()
            metrics = compute_metrics(m, args.eval_lang_retrieval, args.eval_msrvtt)
            print_computed_metrics(metrics)

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

        batch_loss,queue_v,queue_a,queue_t,use_the_queue = TrainOneBatch(net, optimizer, sample_batch, loss_op,queue_v,queue_a,queue_t,use_the_queue,iteration, save_epoch, apex)
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
    for param_group in optimizer.param_groups:
        param_group['lr'] *= args.lr_decay
    if args.checkpoint_dir != '':
        path = os.path.join(args.checkpoint_dir, 'e{}.pth'.format(save_epoch))
        net.module.save_checkpoint(path)
        if args.apex_level == 1:
            amp_checkpoint = {'net': net.module.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'amp': amp.state_dict()}
            th.save(amp_checkpoint, os.path.join(args.checkpoint_dir, 'amp_checkpoint.pt'))


if args.eval_youcook:
    Eval_retrieval(net, dataloader_val, 'YouCook2')
if args.eval_msrvtt:
    Eval_retrieval(net, dataloader_msrvtt, 'MSR-VTT')
if args.eval_lsmdc:
    Eval_retrieval(net, dataloader_lsmdc, 'LSMDC')