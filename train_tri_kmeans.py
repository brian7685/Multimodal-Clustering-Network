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
from youtube_dataloader import Youtube_DataLoader
from youcook_dataloader import Youcook_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from lsmdc_dataloader import LSMDC_DataLoader
from model_kmeans_ICCV import Net
from loss import MMS_loss
from metrics import compute_metrics, print_computed_metrics, AverageMeter
from datetime import datetime
import math
from torch.optim.lr_scheduler import LambdaLR
from fast_pytorch_kmeans import KMeans
import torch.nn.functional as F
import time
random.seed(time.time())

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
    recon=args.recon,
    withMLP=args.withMLP,
    recon_size=args.recon_size
)

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

def update_queue(queue,use_the_queue,fuse):
    bs = int(4096/2)
    fuse2 = fuse.detach()
    fuse2 = fuse2.view(-1, 32, fuse2.shape[-1])
    fuse2 = fuse2[:,:16,:]
    fuse2 = fuse2.reshape(-1, fuse2.shape[-1])
    out = fuse.detach()
    if queue is not None:  # no queue in first round
        if use_the_queue or not th.all(queue[ -1, :] == 0):  # queue[2,3840,128] if never use the queue or the queue is not full
            use_the_queue = True
            # print('use queue')
            out = th.cat((queue,fuse.detach()))  # queue [1920*128] w_t [128*3000] = 1920*3000 out [32*3000] 1952*3000

            #print('out size',out.shape)
        # fill the queue
        queue[ bs:] = queue[ :-bs].clone()  # move 0-6 to 1-7 place
        queue[:bs] = fuse2
    return queue,out,use_the_queue

def cluster_contrast(fushed,centroid,labels,bs):
    S = th.matmul(fushed, centroid.t())

    target = th.zeros(bs,centroid.shape[0]).to(S.device)

    target[range(target.shape[0]), labels] = 1

    S = S - target * (0.001)

    if args.nce==0:
        I2C_loss = F.nll_loss(F.log_softmax(S, dim=1), labels)

    else:
        S = S.view(S.shape[0], S.shape[1], -1)
        nominator = S * target[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = th.logsumexp(nominator, dim=1)
        denominator = S.view(S.shape[0], -1)
        denominator = th.logsumexp(denominator, dim=1)
        I2C_loss = th.mean(denominator - nominator)

    return I2C_loss

def TrainOneBatch(model, opt, data, loss_fun,queue_v,use_the_queue, scheduler, epoch,i_batch, centroid, apex=False):


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
            if args.tri_modal_fuse:
                audio_text, video = model(video, audio, nframes, text)
                sim_audiotext_video = th.matmul(audio_text, video.t())
                loss = loss_fun(sim_audiotext_video)
            else:
                if args.recon:
                    if args.withMLP == 0:
                        audio, video, text, recon_loss = model(video, audio, nframes, text)
                    else:
                        audio, video, text, out_a, out_v, out_t, recon_loss = model(video, audio, nframes, text)
                    recon_w = 50
                    recon_loss = th.mean(recon_loss) * recon_w
                else:
                    if args.withMLP == 0:
                        audio, video, text = model(video, audio, nframes, text)
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)
                # save features B x Pair x D

                if args.withMLP == 0:
                    video_out = video
                    audio_out = audio
                    text_out = text
                else:
                    video_out = out_v
                    audio_out = out_a
                    text_out = out_t

                #pdb.set_trace()
                if args.rand ==0:
                    fushed = (video_out + audio_out + text_out) / 3
                    if args.no_audio:
                        fushed = (video_out + text_out) / 2
                    elif args.no_video:
                        fushed = (  audio_out + text_out) / 2

                if args.joint==1:
                    video_out = audio_out = text_out = fushed

                sim_audio_video = th.matmul(audio, video.t())
                sim_audio_text = th.matmul(audio, text.t())
                sim_text_video = th.matmul(text, video.t())

                if args.no_audio:
                    loss = loss_fun(sim_text_video)
                elif args.no_video:
                    loss = loss_fun(sim_audio_text)
                else:
                    loss = loss_fun(sim_text_video) + loss_fun(sim_audio_text) + loss_fun(sim_audio_video)

                if args.kmeans==1:
                    if args.use_queue==1:
                        queue_v,out,use_the_queue = update_queue(queue_v,use_the_queue,fushed.detach())
                        kmeans = KMeans(n_clusters=args.cluster_size, mode='cosine')#, verbose=1)

                        if args.fastC==1:
                            if i_batch%(int(args.queue_size))==0:
                                labels = kmeans.fit_predict(out,centroid)
                                centroid = kmeans.centroids
                            else:
                                labels = kmeans.max_sim(a=out, b=centroid)[1]

                        else:
                            labels = kmeans.fit_predict(out)
                            centroid = kmeans.centroids

                    else:
                        kmeans = KMeans(n_clusters=args.cluster_size, mode='cosine', verbose=1)
                        labels = kmeans.fit_predict(fushed)

                    if args.mean==1:
                        loss_val = cluster_contrast(fushed,centroid,labels[-bs:],bs)
                    else:
                        if args.no_audio:
                            loss_val = cluster_contrast(video_out, centroid, labels[-bs:], bs) + \
                                       cluster_contrast(text_out, centroid, labels[-bs:], bs)
                        elif args.no_video:
                            loss_val = cluster_contrast(audio_out, centroid, labels[-bs:], bs) + \
                                       cluster_contrast(text_out, centroid, labels[-bs:], bs)
                        else:
                            loss_val = cluster_contrast(video_out, centroid,labels[-bs:],bs) + \
                                       cluster_contrast(audio_out, centroid, labels[-bs:], bs) + \
                                       cluster_contrast(text_out, centroid, labels[-bs:], bs)
                        loss_val = loss_val / 3

                    if i_batch % 100==0:
                        print('loss: ', loss)
                        print('loss_val: ', loss_val)

                    loss+=loss_val*args.clu_lamb

                if args.recon:
                    if i_batch % 100 == 0:
                        print('recon_loss: ', recon_loss)
                    loss += recon_loss


        else:
            audio, video = model(video, audio, nframes)
            sim_matrix = th.matmul(audio, video.t())
            loss = loss_fun(sim_matrix)
    if apex:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    opt.step()
    scheduler.step()
    return loss.item(),queue_v, use_the_queue, centroid

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
                else:
                    if args.recon==1:
                        audio, video, text, recon_loss = model(video, audio, nframes, text)
                    else:
                        audio, video, text, out_a, out_v, out_t = model(video, audio, nframes, text)

                    if args.eval_msrvtt==1:
                        audio_video=video+audio
                    else:
                        audio_video = video+audio
                    m = th.matmul(text, audio_video.t()).cpu().detach().numpy()
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


for epoch in range(args.epochs):
    save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
        else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1


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

    if args.withMLP==1:
        e_size = args.project_dim
    else:
        e_size = args.embd_dim
    queue_l = int(args.queue_size)*(int(args.n_pair/2))*args.batch_size
    if args.use_queue==1 and epoch >= args.start_queue and queue_v is None:  # will start at epoch 15
        queue_v = th.zeros(
            queue_l,
            e_size,
        ).cuda()

    save_epoch = epoch + 1 if args.pretrain_path == '' or 'e' not in args.pretrain_path[-7:-5] \
        else int(args.pretrain_path.split('/')[-1].strip('e.pth')) + epoch + 1
    use_the_queue = False
    centroid = None
    for i_batch, sample_batch in enumerate(tqdm(dataloader)):
        data_load_time = time.time() - end_time
        data_time.update(data_load_time)

        iteration = epoch * len(dataloader) + i_batch  # 0

        batch_loss,queue_v,use_the_queue,centroid = TrainOneBatch(net, optimizer, sample_batch, loss_op,queue_v,use_the_queue,scheduler, save_epoch,i_batch,centroid, apex)
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