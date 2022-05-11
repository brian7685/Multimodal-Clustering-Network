import argparse

def get_args(description='Youtube-Text-Video'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--train_csv',
        type=str,
        default='data/HowTo100M_1166_videopaths.txt',
        #default='/home/brian27/nobackup/data/howto100m/HowTo100M_1166_videopaths.txt',
        help='train csv')
    parser.add_argument(
        '--features_path',
        type=str,
        default='parsed_videos/',
        help='path for visual features (2D, 3D) visual features')
    parser.add_argument(
        '--features_path_audio',
        type=str,
        default='',
        help='path for audio files (defaults to --features_path)')
    parser.add_argument(
        '--caption_path',
        type=str,
        default='data/caption.pickle',
        help='HowTo100M caption pickle file path')
    parser.add_argument(
        '--word2vec_path',
        type=str,
        default='data/GoogleNews-vectors-negative300.bin',
        help='word embedding path')
    parser.add_argument(
        '--pretrain_path',
        type=str,
        default='',
        help='pre train model path')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='',
        help='checkpoint model folder')
    parser.add_argument('--eval_lang_retrieval', type=int, default=0,
                    help='if 1, eval language retrieval instead of video retrieval')
    parser.add_argument('--tri_modal', type=int, default=0,
                            help='use vision, speech, and text')
    parser.add_argument('--tri_modal_fuse', type=int, default=0,
                            help='use speech and text features (tri_modal must be 1)')
    parser.add_argument('--fuse_videoaudio_additive', type=int, default=0,
                            help='eval T->A+V with tri-modal modal \
                                  set tri_modal=1, tri_modal_fuse=0')
    parser.add_argument('--loss', type=int, default=0,
                                help='0 for Masked Margin Softmax (MMS) loss')
    parser.add_argument('--apex_level', type=int, default=0,
                                help='Apex (mixed precision) level: chose 0 for none, 1 for O1.')
    parser.add_argument('--random_audio_windows', type=int, default=1,
                                help='1 to use random audio windows, 0 to use HowTo100M ASR clips')
    parser.add_argument('--howto_audio_frames', type=int, default=1024,
                            help='number of frames to use for loading howto100m audio')
    parser.add_argument('--youcook_num_frames_multiplier', type=int, default=5,
                                help='use 1024 * x audio frames for youcook2')
    parser.add_argument('--msrvtt_num_frames_multiplier', type=int, default=3,
                                help='use 1024 * x audio frames for msrvtt')
    parser.add_argument('--lsmdc_num_frames_multiplier', type=int, default=3,
                                help='use 1024 * x audio frames for lsmdc')
    parser.add_argument('--num_thread_reader', type=int, default=1,
                                help='')
    parser.add_argument('--embd_dim', type=int, default=2048,
                                help='embedding dim')
    parser.add_argument('--lr', type=float, default=0.0001,
                                help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20,
                                help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256,
                                help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500,
                                help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9,
                                help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=200,
                                help='Information display frequence')
    parser.add_argument('--feature_dim', type=int, default=4096,
                                help='video feature dimension')
    parser.add_argument('--we_dim', type=int, default=300,
                                help='word embedding dimension')
    parser.add_argument('--seed', type=int, default=1,
                                help='random seed')
    parser.add_argument('--verbose', type=int, default=1,
                                help='')
    parser.add_argument('--max_words', type=int, default=20,
                                help='')
    parser.add_argument('--min_words', type=int, default=0,
                                help='')
    parser.add_argument('--feature_framerate', type=int, default=1,
                                help='')
    parser.add_argument('--min_time', type=float, default=5.0,
                                help='Gather small clips')
    parser.add_argument('--n_pair', type=int, default=1,
                                help='Number of video clips to use per video')
    parser.add_argument('--lsmdc', type=int, default=0,
                                help='Train on LSDMC data')
    parser.add_argument('--youcook', type=int, default=0,
                                help='Train on YouCook2 data')
    parser.add_argument('--msrvtt', type=int, default=0,
                                help='Train on MSRVTT data')
    parser.add_argument('--eval_lsmdc', type=int, default=0,
                                help='Evaluate on LSMDC data')
    parser.add_argument('--eval_msrvtt', type=int, default=0,
                                help='Evaluate on MSRVTT data')
    parser.add_argument('--eval_youcook', type=int, default=0,
                                help='Evaluate on YouCook2 data')
    parser.add_argument('--eval_ucf', type=int, default=0,
                        help='Evaluate on UCF-101 data')
    parser.add_argument('--eval_hmdb', type=int, default=0,
                        help='Evaluate on HMDB data')
    parser.add_argument('--eval_cross', type=int, default=0,
                        help='Evaluate on CrossTask data')
    parser.add_argument('--eval_how', type=int, default=0,
                        help='Evaluate on how2 data')
    parser.add_argument('--sentence_dim', type=int, default=-1,
                                help='sentence dimension')
    parser.add_argument('--cluster', type=int, default=0,
                        help='cluster loss')
    parser.add_argument('--queue_size', type=int, default=3,
                        help='queue size')
    parser.add_argument('--start_queue', type=int, default=0,
                        help='start_queue')
    parser.add_argument('--start_cluster', type=int, default=0,
                        help='start_cluster')
    parser.add_argument('--num_candidates', type=int, default=1,
                        help='num candidates for MILNCE loss')
    parser.add_argument('--use_queue', type=int, default=0,
                        help='use_queue')
    parser.add_argument('--cluster_size', type=int, default=256,
                        help='cluster_size')
    parser.add_argument('--layer', type=int, default=0,
                        help='classification layer')
    parser.add_argument('--soft_label', type=int, default=0,
                        help='soft_label')
    parser.add_argument('--multi_cluster', type=int, default=0,
                        help='multi_cluster')
    parser.add_argument('--pure_cluster', type=int, default=0,
                        help='pure_cluster')
    parser.add_argument('--project', type=int, default=0,
                        help='project')
    parser.add_argument('--proto_nce', type=int, default=0,
                        help='proto_nce')
    parser.add_argument('--switch_loss_h', type=int, default=0,
                        help='switch_loss_h')
    parser.add_argument('--switch_loss_s', type=int, default=0,
                        help='switch_loss_s')
    parser.add_argument('--self_prediction', type=int, default=0,
                        help='self_prediction')
    parser.add_argument('--soft_contrast', type=int, default=0,
                        help='soft_contrast')
    parser.add_argument('--soft_contrast_only', type=int, default=0,
                        help='soft_contrast_only')
    parser.add_argument('--nce', type=int, default=0,
                        help='nce')
    parser.add_argument('--nce_only', type=int, default=0,
                        help='nce_only')
    parser.add_argument('--pseudo_contrast', type=int, default=0,
                        help='pseudo_contrast')
    parser.add_argument('--cooperative', type=int, default=0,
                        help='cooperative')
    parser.add_argument('--project_dim', type=int, default=6000,
                        help='project_dim')
    parser.add_argument('--no_audio', type=int, default=0,
                        help='no_audio')
    parser.add_argument('--no_va', type=int, default=0,
                        help='no_va')
    parser.add_argument('--rand', type=int, default=0,
                        help='random drop')
    parser.add_argument('--joint', type=int, default=0,
                        help='joint cluster')
    parser.add_argument('--kmeans', type=int, default=0,
                        help='kmeans cluster')
    parser.add_argument('--fastC', type=int, default=0,
                        help='fast cluster')
    parser.add_argument('--withMLP', type=int, default=0,
                        help='withMLP cluster')
    parser.add_argument('--recon', type=int, default=0,
                        help='recon ')
    parser.add_argument('--mms', type=int, default=0,
                        help='mms ')
    parser.add_argument('--mean', type=int, default=0,
                        help='mean ')
    parser.add_argument('--lamb', type=float, default=0.5,
                        help='lambda ')
    parser.add_argument('--tri_loss', type=int, default=0,
                        help='tri_loss ')
    parser.add_argument('--recon_size', type=int, default=768,
                        help='recon_size ')
    parser.add_argument('--clu_lamb', type=int, default=1,
                        help='clu_lamb ')
    parser.add_argument('--noC', type=int, default=0,
                        help='noC ')
    parser.add_argument('--cos', type=int, default=1,
                        help='cos ')
    parser.add_argument("--base_lr", default=4.8, type=float, help="base learning rate")
    parser.add_argument("--final_lr", type=float, default=0, help="final learning rate")
    parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                        help="freeze the prototypes during this many iterations from the start")
    parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
    parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
    parser.add_argument("--start_warmup", default=0, type=float,
                        help="initial warmup learning rate")
    parser.add_argument('--warmup_steps', type=int, default=5000,
                        help='')
    parser.add_argument(
        '--youcook_train_path',
        type=str,
        default='data/youcook_train_audio.pkl',
        help='')
    parser.add_argument(
        '--youcook_val_path',
        type=str,
        default='data/youcook_val_audio.pkl',
        help='')
    parser.add_argument(
        '--msrvtt_test_path',
        type=str,
        default='data/msrvtt_jsfusion_test.pkl',
        help='')
    parser.add_argument(
        '--msrvtt_train_path',
        type=str,
        default='data/msrvtt_train.pkl',
        help='')
    parser.add_argument(
        '--lsmdc_test_path',
        type=str,
        default='data/lsmdc_test.pkl',
        help='')
    parser.add_argument(
        '--lsmdc_train_path',
        type=str,
        default='data/lsmdc_train.pkl',
        help='')
    parser.add_argument(
        '--ucf_test_path',
        type=str,
        default='data/UCF101_data.pkl',
        help='')
    parser.add_argument(
        '--hmdb_test_path',
        type=str,
        default='data/HMDB_data.pkl',
        help='')
    args = parser.parse_args()
    return args