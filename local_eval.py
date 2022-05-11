import numpy as np
from metrics import compute_metrics, print_computed_metrics
import pickle
import torch


eval_lang_retrieval = 0
eval_msrvtt = 1

data = pickle.load(open('temp_data/MSR-VTT.pkl', 'rb'))
#data = pickle.load(open('temp_data/YouCook2.pkl', 'rb'))

text = data['text']
video = data['video']
audio = data['audio']

text2 = data['out_t']
video2 = data['out_v']
audio2 = data['out_a']

text3 = data['out_t2']
video3 = data['out_v2']
audio3 = data['out_a2']

#m = np.matmul(text, video.T) #+ np.matmul(text2, video2.T)
#m = np.matmul(text, (video+audio).T) #+ np.matmul(text2, video2.T)#+ np.matmul(text, audio.T)
m = np.matmul(text, (video).T)# + np.matmul(text, (audio).T)
#m = np.matmul(text, (audio).T)

metrics = compute_metrics(m, eval_lang_retrieval, eval_msrvtt)
print('Combined Space')
print_computed_metrics(metrics)

def norm(mat, axis=-1):
    return np.sqrt(np.sum(mat**2, axis=axis, keepdims=True) + 1e-9)


def softmax(x, axis=-1):
    return np.exp(x)/np.sum(np.exp(x)+1e-12, axis=axis, keepdims=True)

#text2 = text3#softmax(text2*10)
#video2 = video3#softmax(video2*10)

m = np.matmul(text2, (video2).T)# + np.matmul(text2, (audio2).T)

metrics = compute_metrics(m, eval_lang_retrieval, eval_msrvtt)
print('Dot Product on Embedding 2')
print_computed_metrics(metrics)

text2 = softmax(text2*10)
video2 = softmax(video2*10)
m = np.matmul(text2, (video2).T)# + np.matmul(text2, (audio2).T)
metrics = compute_metrics(m, eval_lang_retrieval, eval_msrvtt)
print('Dot Product on softmax Embedding 2 x10 temp')
print_computed_metrics(metrics)


text2 = text3#softmax(text2*10)
video2 = text3#softmax(video2*10)
m = np.matmul(text3, (video3).T)# + np.matmul(text2, (audio2).T)
metrics = compute_metrics(m, eval_lang_retrieval, eval_msrvtt)
print('Dot Product on normalized Embedding')
print_computed_metrics(metrics)

exit()
m = torch.zeros((text2.shape[0], video2.shape[0]))

text2 = torch.from_numpy(text2)
video2 = torch.from_numpy(video2)


for i, v in enumerate(video2):
    diff = (text2 - torch.unsqueeze(v, 0)) ** 2
    diff = torch.sum(diff, -1)
    m[:, i] = 0-diff

metrics = compute_metrics(m, eval_lang_retrieval, eval_msrvtt)
print('Euclidian Distance Embedding 2')
print_computed_metrics(metrics)

