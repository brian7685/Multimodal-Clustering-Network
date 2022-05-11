from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch as th
from torch.utils.data import DataLoader
from args import get_args
import numpy as np
from dp.dp import dp
from tqdm import tqdm
from sklearn.metrics import average_precision_score
from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True)
import torch.nn as nn
from metrics import compute_metrics, print_computed_metrics
from gensim.models.keyedvectors import KeyedVectors
import pickle
import glob
from lsmdc_dataloader import LSMDC_DataLoader
from msrvtt_dataloader import MSRVTT_DataLoader
from youcook_dataloader import Youcook_DataLoader
from cvpr19_dataloader import CVPR19_DataLoader
from mining_dataloader import Mining_DataLoader
import pprint

#th.backends.cudnn.enabled = False

pp = pprint.PrettyPrinter(indent=4)

args = get_args()
if args.verbose:
    print(args)

assert args.pretrain_path != '', 'Need to specify pretrain_path argument'



print('Loading word vectors: {}'.format(args.word2vec_path))
we = KeyedVectors.load_word2vec_format(args.word2vec_path, binary=True)
print('done')

if args.save_feature==1:
    step_path = 'step_all.json'
else:
    step_path = 'step.json'

if args.cross:
    cross_testset = CVPR19_DataLoader(
        #csv='vids_27.csv',
        csv='/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/cvpr19_test.csv',
        features_path='vids_feature',
        #features_path = '/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/features_2d',
        #features_path_3D = '/nobackup/users/brian27/CrossTask/howto100m_crosstask_eval/features_3d',
        annot_path = 'anno', #'/nobackup/users/brian27/CrossTask/crosstask_release/Y-1',
        steps_path = step_path,
        audio_path = 'audio_feature',#'/home/brian27/nobackup/CrossTask/audio_feature_new',#
        annot_path_time='/nobackup/users/brian27/CrossTask/crosstask_release/annotations',
        cook_path = '/home/brian27/nobackup/CrossTask/crosstask_release/cook.txt',
        with_audio = args.with_audio,
        we=we
        #features_path_3D='howto100m_crosstask_eval/features_3d'
    )
if args.mining:
    cross_testset = Mining_DataLoader(
        csv='/nobackup/users/brian27/Weak_YouTube_dataset/mining.csv',
        features_path='/nobackup/users/brian27/Weak_YouTube_dataset/test_new_f',
        annot_path='/nobackup/users/brian27/Weak_YouTube_dataset/anno',
        steps_path='/nobackup/users/brian27/Weak_YouTube_dataset/'+step_path,
        audio_path='/nobackup/users/brian27/Weak_YouTube_dataset/test_new_a_f',  # 'audio_feature',
        we=we
        # features_path_3D='howto100m_crosstask_eval/features_3d'
    )
#print(cross_testset)
dataloader_cross = DataLoader(
    cross_testset,
    batch_size=1,
    num_workers=args.num_thread_reader,
    shuffle=False,
    drop_last=False,
)


#def cvpr19_score(X, steps, model):
def cvpr19_score_a(X, audio, nframes, steps, model):
    #sim_matrix = model.forward(X.cuda(),steps.cuda()).transpose(1,0) #[frame,class]
    #print('video',X.shape)
    #print('audio',audio.shape)
    #print('text',steps.shape)
    if args.v_only==1:
        sim_matrix = model.forward(X, audio, nframes, args.v_only, steps)
        return sim_matrix.transpose(1, 0).detach().cpu().numpy()

    #sim_matrix,s2,s3 = model.forward(X, audio, nframes, steps).transpose(1, 0)
    sim_matrix,s2,s3 = model.forward(X, audio, nframes, args.v_only, steps)  # [frame,class]
    #v,a,t = model.forward(X, audio, nframes, steps)#.transpose(1, 0)  # [frame,class]
    #print('sim_matrix',sim_matrix.shape)
    return sim_matrix.transpose(1, 0).detach().cpu().numpy(),s2.transpose(1, 0).detach().cpu().numpy(),s3.transpose(1, 0).detach().cpu().numpy()
    #return v,a,t

def cvpr19_score(X, steps, model):
    sim_matrix = model.forward(X.cuda(),steps.cuda()).transpose(1,0) #[frame,class]
    #print('video',X.shape)
    #print('audio',audio.shape)
    #print('text',steps.shape)
    #sim_matrix = model.forward(X, audio, nframes, steps).transpose(1, 0)  # [frame,class]
    #print('sim_matrix',sim_matrix.shape)
    return sim_matrix.detach().cpu().numpy()

def cvpr19_predict(scores):
    C = -scores#.cpu().detach().numpy()
    y = np.empty(scores.shape, dtype=np.float32)
    dp(y, C, exactly_one=True) #[frame,class]
    return y

def arg_max_predict(scores):
    y_final = np.zeros((scores.shape[0], scores.shape[1]))
    arg_y = np.argmax(scores, axis=1)
    for i in range(scores.shape[0]):
        y_final[i][arg_y[i]] = 1
    return y_final

def get_recall(y_true, y):
    #return ((y*y_true).sum(axis=1)>0).sum() / (y_true.sum(axis=1)>0).sum()
    if args.recall_frame==0:
        return ((y*y_true).sum(axis=0)>0).sum() / (y_true.sum(axis=0)>0).sum()
    else:
        return ((y * y_true).sum(axis=0) > 0).sum() / (y_true.sum(axis=0) > 0).sum()

def align_eval(model, dataloader, gpu_mode=1):
    print('start cross')
    recalls = {}
    counts = {}
    recalls_m = 0
    counts_m = 0
    task_scores = {}
    task_gt = {}
    for sample in tqdm(dataloader):
        with th.no_grad():

            #print(sample)
            #for sample in batch:



            video = sample['video'].cuda() if gpu_mode else sample['video']
            text = sample['steps'].cuda() if gpu_mode else sample['steps']

            video = video.view(-1, video.shape[-1])
            text = th.squeeze(text)# class x emb
            #n_frame = th.tensor([])
            n_frame = sample['nframes'].cuda()#th.ones(video.shape[0],1)*1#.cuda()
            n_frame = n_frame.view(-1)
            #print(n_frame.shape)

            #print('n_frame',n_frame.shape)
            if args.tri==1:
                audio = sample['audio'].cuda() if gpu_mode else sample['video']
                audio = audio.view(-1, audio.shape[-2], audio.shape[-1])
                #print(audio.shape)
                scores_list = []
                split = 15
                batch_size = 25
                #print(video.shape[0])
                b_s = int(video.shape[0] / batch_size)
                # for i in range(video.shape[0]):
                # video_1 = th.unsqueeze(video[:half],0)
                # audio_1 = th.unsqueeze(audio[:half],0)
                if video.shape[0] < batch_size:
                    if args.v_only==0:
                        scores,s2,s3 = cvpr19_score_a(video, audio, n_frame, text, model)
                    else:
                        scores = cvpr19_score_a(video, audio, n_frame, text, model)
                else:
                    for i in range(b_s):
                        if i == b_s - 1:
                            video_1 = video[batch_size * i:]
                            audio_1 = audio[batch_size * i:]
                            n_frame_1 = n_frame[batch_size * i:]
                        else:
                            video_1 = video[batch_size * i:batch_size * (i + 1)]
                            audio_1 = audio[batch_size * i:batch_size * (i + 1)]
                            n_frame_1 = n_frame[batch_size * i:batch_size * (i + 1)]
                        # text_1 = th.unsqueeze(text[i])
                        if args.v_only==0:
                            scores,s2,s3 = cvpr19_score_a(video_1, audio_1, n_frame_1, text, model)
                        else:
                            scores = cvpr19_score_a(video_1, audio_1, n_frame_1, text, model)
                        scores_list.append(scores)
                    scores = np.vstack(scores_list)
                if args.save_feature==1:

                    scores = th.from_numpy(scores)
                    m = nn.LogSoftmax(dim=1)
                    scores = m(scores).detach().cpu().numpy()
                    #print(scores)
                    method = args.method_name
                    if args.mining == 1:
                        path = 'mining_score_'+method+'/'
                    else:
                        path = 'cross_score_' + method + '/'
                    from pathlib import Path
                    Path(path).mkdir(parents=True, exist_ok=True)

                    file1 = open(path + sample['video_id'][0] + '.probs', 'w')
                    for i in range(scores.shape[0]):
                        for j in range(scores.shape[1]):
                            # for k in range(30):
                            file1.write(str(scores[i][j]) + ' ')
                        file1.write('\n')
                    file1.close()

            else:
                scores = cvpr19_score(video, text, model) #[time,class]
                if args.save_feature == 1:
                    scores = th.from_numpy(scores)
                    m = nn.LogSoftmax(dim=1)
                    scores = m(scores).detach().cpu().numpy()
                    from pathlib import Path
                    path = 'mining_score_ver/'
                    Path(path).mkdir(parents=True, exist_ok=True)
                    file1 = open(path + sample['video_id'][0] + '.probs', 'w')
                    for i in range(scores.shape[0]):
                        for j in range(scores.shape[1]):
                            # for k in range(30):
                            file1.write(str(scores[i][j]) + ' ')
                        file1.write('\n')
                    file1.close()
            #"""
            if args.save_feature == 0:
                if args.recall_frame==0:
                    #scores = np.log(scores)
                    #"""
                    if args.mining==1:
                        m = nn.LogSoftmax(dim=1)
                        #m = nn.LogSigmoid()
                        scores = th.from_numpy(scores)
                        scores = m(scores).detach().cpu().numpy()
                    #"""
                    y = cvpr19_predict(scores) #[time,class]
                else:
                    y = arg_max_predict(scores)
                y_true = th.squeeze(sample['Y']).numpy()

                if args.cross==1:
                    task = sample['task']
                    #y_true = y_true.view(-1, y_true.shape[-1])

                    task = task[0]#.view(-1, task.shape[-1])

                    if task not in recalls:
                        recalls[task] = 0.
                    recalls[task] += get_recall(y_true, y)
                    if task not in counts:
                        counts[task] = 0
                    counts[task] += 1

                    # mAP ----------------------------------------
                    if task not in task_scores:
                        task_scores[task] = []
                        task_gt[task] = []
                    task_scores[task].append(scores)
                    task_gt[task].append(y_true)
                else:
                    recalls_m += get_recall(y_true, y)
                    counts_m += 1

            #if task == '77721':
            #    print('recall:', recalls['77721'])
            #    print('counts:', counts['77721'])
            #    print(sample['video_id'])
            #print(recalls)
            #"""
            # --------------------------------------------
    #"""
    if args.save_feature == 0:
        if args.cross==1:
            recalls = {task: recall / counts[task] for task,recall in recalls.items()}
            # mAP ----------------------------------------
            task_scores = {task: np.concatenate(scores) for task,scores in task_scores.items()}
            task_gt = {task: np.concatenate(y) for task,y in task_gt.items()}
            mAPs = {task: average_precision_score(task_gt[task],scores) for task,scores in task_scores.items()}
            # --------------------------------------------
            #"""
            return recalls, mAPs
        else:
            print(recalls_m/counts_m)
            return recalls_m, None



if args.tri == 0:
    from model import Net
else:
    from model_avl import Net

net = Net(
    embd_dim=args.embd_dim, #2048
    video_dim=args.feature_dim, #4096
    we_dim=args.we_dim,
    ratio=args.ratio,
)

net.eval()
net.cuda()

if args.verbose:
    print('Starting evaluation loop ...')


all_checkpoints = glob.glob(args.pretrain_path)

for c in all_checkpoints:
    print('Eval checkpoint: {}'.format(c))
    print('Loading checkpoint: {}'.format(c))
    net.load_checkpoint(c)

    if args.save_feature == 1:
        align_eval(net, dataloader_cross)
    elif args.save_feature == 0:
        recall, mAPs = align_eval(net, dataloader_cross)

        pp.pprint(recall)
        if args.cross==1:

            pp.pprint(mAPs)
            sum = 0
            count = 0
            sum_c = 0
            count_c = 0
            sum_nc = 0
            count_nc = 0

            cook_set=set()
            file1 = open('/home/brian27/nobackup/CrossTask/crosstask_release/cook.txt')
            for line in file1:
                data = line.strip()
                cook_set.add(data)

            for x,y in recall.items():
                sum+=y
                count+=1
                if x in cook_set:
                    sum_c += y
                    count_c += 1
                else:
                    sum_nc += y
                    count_nc += 1

            print('recall',sum/float(count))
            print('recall cook', sum_c / float(count_c))
            print('recall not cook', sum_nc / float(count_nc))
            sum = 0
            count = 0
            for x,y in mAPs.items():
                sum+=y
                count+=1
            print('mAPs',sum/float(count))
            #"""
