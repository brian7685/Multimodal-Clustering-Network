import pickle
import numpy as np


def recall(mat, gts):
    # mat is of shape (Queries, Targets), where higher=prediction
    # gts is of shape (Queries, )

    predictions = np.argsort(mat, 1)  # (Queries, Targets)

    top_1 = predictions[:, -1]

    recall = np.mean(top_1 == gts)
    print('NN Retrieval R@1:', recall)

    recall_top5 = np.mean([l in p for l, p in zip(gts, predictions[:, -5:])])
    print('NN Retrieval R@5:', recall_top5)

    recall_top10 = np.mean([l in p for l, p in zip(gts, predictions[:, -10:])])
    print('NN Retrieval R@10:', recall_top10)


def evaluate_recall_youcook(text, video_audio, clip_ids, m=None):
    # text is of shape (n_clips, n_feats)
    # video_audio is of shape (n_clips, n_feats)
    # video_ids is a list of length n_clips with all the clip_ids
    full_videos = sorted(list(set([d[:11] for d in clip_ids])))
    print('# Clips', len(clip_ids))
    print('# Videos', len(full_videos))

    n_clips = len(clip_ids)
    n_vids = len(full_videos)
    clip_to_video = []
    [clip_to_video.extend([i for i, x in enumerate(full_videos) if x in clip_id]) for clip_id in clip_ids]
    clip_to_video = np.array(clip_to_video)

    if m is None:
        m = np.matmul(text, video_audio.T)  # (n_clips, n_clips)
    print('Standard Retrieval | single caption -> single clip')
    recall(m, np.arange(m.shape[0]))

    predictions = np.argsort(m, 1)

    video_predictions = clip_to_video[predictions]
    video_gts = clip_to_video[np.arange(len(clip_to_video))]

    print('Retrieval single | single caption -> full video')
    recall_top1 = np.mean(video_predictions[:, -1] == video_gts)
    print('NN Retrieval R@1:', recall_top1)

    recall_top5 = np.mean([l in p for l, p in zip(video_gts, video_predictions[:, -5:])])
    print('NN Retrieval R@5:', recall_top5)

    recall_top10 = np.mean([l in p for l, p in zip(video_gts, video_predictions[:, -10:])])
    print('NN Retrieval R@10:', recall_top10)

    video_inds = [[i for i, x in enumerate(clip_ids) if video in x] for video in full_videos] # list of length n_vids, with the corresponding clip_inds

    video_preds_m = np.stack([np.max(m[:, v], axis=1) for v in video_inds], 1)  # (n_clips, n_vids)
    video_preds_m2 = np.stack([np.mean(video_preds_m[v, :], axis=0) for v in video_inds], 0)  # (n_vids, n_vids)

    print('Retrieval single | full caption -> full video | for each caption get max prediction over a video, then average over all captions of a video.')
    recall(video_preds_m2, np.arange(n_vids))

    corr_preds = []
    for video_id in range(len(full_videos)):
        vid_i_m = video_preds_m[video_gts == video_id]
        vid_i_pred = np.argsort(vid_i_m, 1)
        prs = []
        for i in [1, 5, 10]:
            top_i_preds = vid_i_pred[:, -i:]
            unique_ids, counts = np.unique(top_i_preds, return_counts=True)
            id_pred = unique_ids[np.argsort(counts)[-i:]]
            #print(id_pred)
            prs.append(video_id in id_pred)
        corr_preds.append(prs)

    t1, t5, t10 = zip(*corr_preds)
    print('Retrieval single | full caption -> full video | for each caption get top_k video predictions, then get sorted majority vote for final top_k predictions.')
    print('NN Retrieval R@1:', np.mean(t1))
    print('NN Retrieval R@5:', np.mean(t5))
    print('NN Retrieval R@10:', np.mean(t10))

    corr_preds = []
    for video_id in range(len(full_videos)):
        vid_i_m = m[video_gts == video_id]
        vid_i_pred = clip_to_video[np.argsort(vid_i_m, 1)]

        prs = []
        for i in [1, 5, 10]:
            top_i_preds = vid_i_pred[:, -i:]
            unique_ids, counts = np.unique(top_i_preds, return_counts=True)
            id_pred = unique_ids[np.argsort(counts)[-i:]]
            prs.append(video_id in id_pred)
        corr_preds.append(prs)

    t1, t5, t10 = zip(*corr_preds)
    print('Retrieval single | full caption -> full video | for each caption get top_k clip predictions, then get sorted majority vote for final top_k predictions.')
    print('NN Retrieval R@1:', np.mean(t1))
    print('NN Retrieval R@5:', np.mean(t5))
    print('NN Retrieval R@10:', np.mean(t10))

#data = pickle.load(open('temp_data/YouCook2.pkl', 'rb'))
#print(data.keys())
#evaluate_recall(data['text'], data['audio']+data['video'], data['video_id'])

