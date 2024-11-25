from statistics import median
import yamlargparse
import os


project = ''
tag = '02' 
train_name = '_train'
adapt_name = '_adapt'
infer_name = '_infer'
seed= 777
wandb_group = ''
wandb_entity = ""
wandb_api_key = ''
neptune_token= ""
neptune_user= ""
train_cuda_visible_devices = '0,1'
test_cuda_visible_devices = '1'
shuffle=False

def get_args():
    train_args = train_argument()
    save_spkv_lab_args = save_spkv_lab_argument()
    adapt_args = adapt_argument()
    infer_args = infer_argument()
    return train_args, save_spkv_lab_args, adapt_args, infer_args


def train_argument():
    cuda_visible_devices = train_cuda_visible_devices
    train_data_dir = '/workspace/EEND/DB/data/swb_sre_tr_ns3_beta10_100000'
    valid_data_dir = '/workspace/EEND/DB/data/swb_sre_cv_ns3_beta10_500'
    preprocess_trial = 'featlab_chunk_indices.txt'
    preprocess_dir = 'data'
    preprocess_flag = '.done'
    model_save_dir = f'/results/{project}/{tag}/{train_name}/model'
    model_filename = "transformer{}.th"
    avg_model_filename = "avg_trans.th"
    spkv_lab = 'd'
    spk_loss_ratio= 0.1
    spk_count_loss_ratio= 0.2 #0.5
    # spkv_dim= 256
    max_epochs= 100
    start_avg_epoch= 90
    end_avg_epoch= 100
    batchsize= 768
    log_report_num = 50
    input_transform= 'logmel23_mn'
    lr= 1.0
    optimizer_name= 'noam'
    num_speakers= 3
    gradclip= 5
    chunk_size= 150
    num_workers= 8
    hidden_size= 256
    context_size= 7
    subsampling = 10
    frame_size= 200
    frame_shift= 80
    sampling_rate= 8000
    noam_scale= 1.0
    noam_warmup_steps= 25000
    transformer_encoder_n_heads= 8
    transformer_encoder_n_layers= 6
    transformer_encoder_dropout= 0.1
    feature_nj= 100
    batchsize_per_gpu= 16
    test_run= 0
    

    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('--cuda_visible_devices', default=cuda_visible_devices, type=str)
    parser.add_argument('--name', default=train_name, type = str)
    parser.add_argument('--tags',default=tag, type = str)
    parser.add_argument('--project',default=project, type = str)
    parser.add_argument('--train_data_dir', default=train_data_dir, type = str,
                        help='kaldi-style data dir used for training.')
    parser.add_argument('--valid_data_dir', default=valid_data_dir, type = str,
                        help='kaldi-style data dir used for validation.')
    parser.add_argument('--model_save_dir', default=model_save_dir, type = str,
                        help='output model_save_dir which model file will be saved in.')
    parser.add_argument('--model_filename', default=model_filename, type = str)
    parser.add_argument('--avg_model_filename', default=avg_model_filename, type = str)
    parser.add_argument('--preprocess_dir', default=preprocess_dir, type = str)
    parser.add_argument('--preprocess_trial', default=preprocess_trial, type = str)
    parser.add_argument('--preprocess_flag', default=preprocess_flag, type = str)
    parser.add_argument('--spkv_lab', default=spkv_lab,
                        help='file path of speaker vector with label and\
                            speaker ID conversion table for adaptation')

    parser.add_argument('--path_logging', default='/results', type = str)
    parser.add_argument('--path_scripts', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--neptune_user', default=neptune_user, type = str)
    parser.add_argument('--neptune_token', default=neptune_token, type = str)
    parser.add_argument('--wandb_group', default=wandb_group, type = str)
    parser.add_argument('--wandb_entity', default=wandb_entity, type = str)
    parser.add_argument('--wandb_api_key', default=wandb_api_key, type = str)
    parser.add_argument('--model-type', default='Transformer',
                    help='Type of model (Transformer)')
    parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
    ##########################################################################
    parser.add_argument('--max-epochs', default=max_epochs, type=int,
                    help='Max. number of epochs to train')
    parser.add_argument('--start_avg_epoch', default=start_avg_epoch, type=int,
                        help='start number of epochs to average')
    parser.add_argument('--end_avg_epoch', default=end_avg_epoch, type=int,
                        help='end number of epochs to average')
    parser.add_argument('--input-transform', default=input_transform,
                    choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn',
                             'logmel23_mvn', 'logmel23_swn'],
                    help='input transform')
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--shuffle', default=shuffle)
    parser.add_argument('--optimizer-name', default=optimizer_name, type=str)
    parser.add_argument('--num-workers', default=num_workers, type=int)
    parser.add_argument('--num-speakers',default=num_speakers, type=int)
    parser.add_argument('--spk-loss-ratio', default=spk_loss_ratio, type=float)
    parser.add_argument('--spk-count-loss-ratio', default=spk_count_loss_ratio, type=float)
    # parser.add_argument('--spkv-dim', default=spkv_dim, type=int,
    #                     help='dimension of speaker embedding vector')
    parser.add_argument('--log-report-num', default=log_report_num, type=int)
    parser.add_argument('--gradclip', default=gradclip, type=int,
                        help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--chunk-size', default=chunk_size, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--num-frames', default=2000, type=int,
                        help='number of frames in one utterance')
    parser.add_argument('--batchsize', default=batchsize, type=int,
                        help='number of utterances in one batch')
    parser.add_argument('--label-delay', default=0, type=int,
                        help='number of frames delayed from original labels'
                            ' for uni-directional rnn to see in the future')
    parser.add_argument('--hidden-size', default=hidden_size, type=int,
                        help='number of lstm output nodes')
    parser.add_argument('--context-size', default=context_size, type=int)
    parser.add_argument('--subsampling', default=subsampling, type=int)
    parser.add_argument('--frame-size', default=frame_size, type=int)
    parser.add_argument('--frame-shift', default=frame_shift, type=int)
    parser.add_argument('--sampling-rate', default=sampling_rate, type=int)
    parser.add_argument('--noam-scale', default=noam_scale, type=float)
    parser.add_argument('--noam-warmup-steps', default=noam_warmup_steps, type=float)
    parser.add_argument('--transformer-encoder-n-heads', default=transformer_encoder_n_heads, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=transformer_encoder_n_layers, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=transformer_encoder_dropout, type=float)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--seed', default=seed, type=int)

    parser.add_argument('--feature-nj', default=feature_nj, type=int,
                        help='maximum number of subdirectories to store\
                        featlab_XXXXXXXX.npy')
    parser.add_argument('--batchsize-per-gpu', default=batchsize_per_gpu, type=int,
                        help='virtual_minibatch_size in padertorch')
    parser.add_argument('--test-run', default=test_run, type=int, choices=[0, 1],
                        help='padertorch test run switch; 1 is on, 0 is off')
    
    args = parser.parse_args()
    args.gpu_ids = args.cuda_visible_devices.split(',')
    if len(args.gpu_ids) == 0:
        raise Exception('Only GPU env are supported')
    args.world_size = len(args.gpu_ids)
    args.batchsize = args.batchsize // (args.world_size)
    args.num_workers = args.num_workers // args.world_size

    return args
    

def save_spkv_lab_argument():
    cuda_visible_devices = test_cuda_visible_devices
    train_adapt_dir = '/workspace/EEND/DB/data/eval/callhome1_spkall'
    model_save_dir = f'/results/{project}/{tag}/{train_name}/model'
    init_model = f'{model_save_dir}/avg_trans.th'
    save_spkv_lab_dir = f'/results/{project}/{tag}/{train_name}/spkv_lab'
    num_speakers= 3
    # spkv_dim= 256
    hidden_size= 256
    input_transform= 'logmel23_mn'
    chunk_size= 300
    context_size= 7
    subsampling= 10
    sampling_rate= 8000
    frame_size= 200
    frame_shift= 80
    transformer_encoder_n_heads= 8
    transformer_encoder_n_layers= 6


    parser = yamlargparse.ArgumentParser(description='decoding')
    parser.add_argument('--cuda_visible_devices', default=cuda_visible_devices, type=str)
    parser.add_argument('--data_dir', default=train_adapt_dir, help='kaldi-style data dir')
    parser.add_argument('--tags',default=tag, type = str)
    parser.add_argument('--project',default=project, type = str)
    parser.add_argument('--model_file',default=init_model, help='best.nnet')
    parser.add_argument('--out_dir',default=save_spkv_lab_dir, help='output directory.')
    parser.add_argument('--num-speakers', default=num_speakers, type=int)
    parser.add_argument('--shuffle', default=shuffle)
    # parser.add_argument('--spkv-dim', default=spkv_dim, type=int, help='dimension of speaker embedding vector')
    parser.add_argument('--hidden-size', default=hidden_size, type=int)
    parser.add_argument('--input-transform', default=input_transform, choices=['', 'log', 'logmel', 'logmel23', 'logmel23_swn', 'logmel23_mn'],help='input transform')
    parser.add_argument('--chunk-size', default=chunk_size, type=int, help='input is chunked with this size')
    parser.add_argument('--context-size', default=context_size, type=int, help='frame splicing')
    parser.add_argument('--subsampling', default=subsampling, type=int)
    parser.add_argument('--sampling-rate', default=sampling_rate, type=int, help='sampling rate')
    parser.add_argument('--frame-size', default=frame_size, type=int, help='frame size')
    parser.add_argument('--frame-shift', default=frame_shift, type=int, help='frame shift')
    parser.add_argument('--transformer-encoder-n-heads', default=transformer_encoder_n_heads, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=transformer_encoder_n_layers, type=int)

    args = parser.parse_args()
    return args
    
    
def adapt_argument():
    cuda_visible_devices = test_cuda_visible_devices
    train_adapt_dir = '/workspace/EEND/DB/data/eval/callhome1_spkall'
    dev_adapt_dir = '/workspace/EEND/DB/data/eval/callhome2_spk3'
    model_adapt_dir = f'/results/{project}/{tag}/{adapt_name}/model'
    model_filename = "transformer{}.th"
    avg_model_filename = "avg_trans.th"
    init_model_dir = f'/results/{project}/{tag}/{train_name}/model'
    init_model = f'{init_model_dir}/avg_trans.th'
    spkv_lab = f'/results/{project}/{tag}/{train_name}/spkv_lab/spkvec_lab.npz'
    preprocess_trial = 'featlab_chunk_indices.txt'
    preprocess_dir = 'data'
    preprocess_flag = '.done'
    spk_loss_ratio= 0.1
    spk_count_loss_ratio= 0.1
    # spkv_dim= 256
    max_epochs= 50
    start_avg_epoch= 50
    end_avg_epoch= 50
    batchsize= 64
    log_report_num = 10
    input_transform= 'logmel23_mn'
    lr= 1e-5
    optimizer_name= 'adam'
    num_speakers= 3
    gradclip= 5
    chunk_size= 300
    num_workers= 8
    hidden_size= 256
    context_size= 7
    subsampling= 10
    frame_size= 200
    frame_shift= 80
    sampling_rate= 8000
    noam_scale= 1.0
    noam_warmup_steps= 25000
    transformer_encoder_n_heads= 8
    transformer_encoder_n_layers= 6
    transformer_encoder_dropout= 0.1
    feature_nj= 100
    batchsize_per_gpu= 8
    test_run= 1
    


    parser = yamlargparse.ArgumentParser(description='EEND training')
    parser.add_argument('--cuda_visible_devices', default=cuda_visible_devices, type=str)
    parser.add_argument('--name',default=adapt_name, type = str)
    parser.add_argument('--tags',default=tag, type = str)
    parser.add_argument('--project',default=project, type = str)
    parser.add_argument('--train_data_dir',default=train_adapt_dir, help='kaldi-style data dir used for training.')
    parser.add_argument('--valid_data_dir',default=dev_adapt_dir, help='kaldi-style data dir used for validation.')
    parser.add_argument('--model_save_dir',default=model_adapt_dir, type = str, help='output model_save_dirdirectory which model file will be saved in.')
    parser.add_argument('--model_filename', default=model_filename, type = str)
    parser.add_argument('--avg_model_filename', default=avg_model_filename, type = str)
    parser.add_argument('--preprocess_dir', default=preprocess_dir, type = str)
    parser.add_argument('--preprocess_trial', default=preprocess_trial, type = str)
    parser.add_argument('--preprocess_flag', default=preprocess_flag, type = str)
    parser.add_argument('--spkv-lab', default=spkv_lab, help='file path of speaker vector with label and speaker ID conversion table for adaptation')
    parser.add_argument('--path_logging', type = str, default='/results')
    parser.add_argument('--path_scripts', default=os.path.dirname(os.path.realpath(__file__)))
    parser.add_argument('--neptune_user', default=neptune_user, type = str)
    parser.add_argument('--neptune_token', default=neptune_token, type = str)
    parser.add_argument('--wandb_group', default=wandb_group, type = str)
    parser.add_argument('--wandb_entity', default=wandb_entity, type = str)
    parser.add_argument('--wandb_api_key', default=wandb_api_key, type = str)
    parser.add_argument('--model-type', default='Transformer', help='Type of model (Transformer)')
    parser.add_argument('--initmodel', '-m', default=init_model, help='Initialize the model from given file')
    parser.add_argument('--resume', '-r', default='', help='Resume the optimization from snapshot')
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--max-epochs', default=max_epochs, type=int, help='Max. number of epochs to train')
    parser.add_argument('--start_avg_epoch', default=start_avg_epoch, type=int, help='start number of epochs to average')
    parser.add_argument('--end_avg_epoch', default=end_avg_epoch, type=int, help='end number of epochs to average')
    parser.add_argument('--input-transform', default=input_transform, choices=['', 'log', 'logmel', 'logmel23', 'logmel23_mn','logmel23_mvn', 'logmel23_swn'],help='input transform')
    parser.add_argument('--lr', default=lr, type=float)
    parser.add_argument('--optimizer-name', default=optimizer_name, type=str)
    parser.add_argument('--num-workers', default=num_workers, type=int)
    parser.add_argument('--num-speakers',default= num_speakers, type=int)
    parser.add_argument('--spk-loss-ratio', default=spk_loss_ratio, type=float)
    parser.add_argument('--spk-count-loss-ratio', default=spk_count_loss_ratio, type=float)
    # parser.add_argument('--spkv-dim', default=spkv_dim, type=int, help='dimension of speaker embedding vector')
    parser.add_argument('--shuffle', default=shuffle)
    parser.add_argument('--log-report-num', default=log_report_num, type=int)
    parser.add_argument('--gradclip', default=gradclip, type=int, help='gradient clipping. if < 0, no clipping')
    parser.add_argument('--chunk-size', default=chunk_size, type=int, help='number of frames in one utterance')
    parser.add_argument('--num-frames', default=2000, type=int, help='number of frames in one utterance')
    parser.add_argument('--batchsize', default=batchsize, type=int, help='number of utterances in one batch')
    parser.add_argument('--label-delay', default=0, type=int, help='number of frames delayed from original labels for uni-directional rnn to see in the future')
    parser.add_argument('--hidden-size', default=hidden_size, type=int, help='number of lstm output nodes')
    parser.add_argument('--context-size', default=context_size, type=int)
    parser.add_argument('--subsampling', default=subsampling, type=int)
    parser.add_argument('--frame-size', default=frame_size, type=int)
    parser.add_argument('--frame-shift', default=frame_shift, type=int)
    parser.add_argument('--sampling-rate', default=sampling_rate, type=int)
    parser.add_argument('--noam-scale', default=noam_scale, type=float)
    parser.add_argument('--noam-warmup-steps', default=noam_warmup_steps, type=float)
    parser.add_argument('--transformer-encoder-n-heads', default=transformer_encoder_n_heads, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=transformer_encoder_n_layers, type=int)
    parser.add_argument('--transformer-encoder-dropout', default=transformer_encoder_dropout, type=float)
    parser.add_argument('--gradient-accumulation-steps', default=1, type=int)
    parser.add_argument('--seed', default=seed, type=int)
    parser.add_argument('--feature-nj', default=feature_nj, type=int, help='maximum number of subdirectories to store featlab_XXXXXXXX.npy')
    parser.add_argument('--batchsize-per-gpu', default=batchsize_per_gpu, type=int, help='virtual_minibatch_size in padertorch')
    parser.add_argument('--test-run', default=test_run, type=int, choices=[0, 1], help='padertorch test run switch; 1 is on, 0 is off')
    
    args = parser.parse_args()
    args.gpu_ids = args.cuda_visible_devices.split(',')
    if len(args.gpu_ids) == 0:
        raise Exception('Only GPU env are supported')
    args.world_size = len(args.gpu_ids)
    args.batchsize = args.batchsize // (args.world_size)
    args.num_workers = args.num_workers // args.world_size

    return args


def infer_argument():
    cuda_visible_devices = test_cuda_visible_devices
    test_dir = '/workspace/EEND/DB/data/eval/callhome2_spk{}'
    visual_sim_dir = '/workspace/EEND/DB/data/swb_sre_cv_ns3_beta10_500'
    visual_ch_dir = '/workspace/EEND/DB/data/eval/callhome2_spk3'
    infer_dir = f'/results/{project}/{tag}/{infer_name}' 
    out_dir = f'{infer_dir}/output' 
    score_dir = f'{infer_dir}/score'+'{}'
    test_model = f'/results/{project}/{tag}/{adapt_name}/model/avg_trans.th'
    est_nspk= 0
    num_speakers= 3
    # spkv_dim= 256
    hidden_size= 256
    input_transform= 'logmel23_mn'
    chunk_size= 300
    context_size= 7
    subsampling= 10
    sampling_rate= 8000
    frame_size= 200
    frame_shift= 80
    transformer_encoder_n_heads= 8
    transformer_encoder_n_layers= 6
    sil_spk_th= 0.05
    ahc_dis_th= 1.4
    clink_dis= 1e+4
    threshold = '0.35,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.52,0.54,0.56,0.58,0.6,0.62,0.65'
    median = '1,3,5,7,9,11'
    collar = 0.25

    parser = yamlargparse.ArgumentParser(description='decoding')
    parser.add_argument('--cuda_visible_devices', default=cuda_visible_devices, type=str)
    parser.add_argument('--data_dir',default=test_dir, help='kaldi-style data dir')
    parser.add_argument('--visual_sim_dir',default=visual_sim_dir, help='kaldi-style data dir')
    parser.add_argument('--visual_ch_dir',default=visual_ch_dir, help='kaldi-style data dir')
    parser.add_argument('--model_file',default=test_model, help='best.nnet')
    parser.add_argument('--infer_dir',default=infer_dir, help='inference directory.')
    parser.add_argument('--out_dir',default=out_dir, help='output directory.')
    parser.add_argument('--score_dir',default=score_dir, help='score directory.')
    
    # The following arguments are set in conf/infer_est_nspk{0,1}.yaml
    parser.add_argument('--est-nspk', default=est_nspk, type=int, choices=[0, 1], help='At clustering stage, --est-nspk 0 means that oracle number of speakers is used, --est-nspk 1 means estimating numboer of speakers')
    parser.add_argument('--num-speakers', default=num_speakers, type=int)
    # parser.add_argument('--spkv-dim', default=spkv_dim, type=int, help='dimension of speaker embedding vector')
    parser.add_argument('--shuffle', default=shuffle)
    parser.add_argument('--hidden-size', default=hidden_size, type=int)
    parser.add_argument('--input-transform', default=input_transform, choices=['', 'log', 'logmel', 'logmel23', 'logmel23_swn', 'logmel23_mn'], help='input transform')
    parser.add_argument('--chunk-size', default=chunk_size, type=int, help='input is chunked with this size')
    parser.add_argument('--context-size', default=context_size, type=int, help='frame splicing')
    parser.add_argument('--subsampling', default=subsampling, type=int)
    parser.add_argument('--sampling-rate', default=sampling_rate, type=int, help='sampling rate')
    parser.add_argument('--frame-size', default=frame_size, type=int, help='frame size')
    parser.add_argument('--frame-shift', default=frame_shift, type=int, help='frame shift')
    parser.add_argument('--transformer-encoder-n-heads', default=transformer_encoder_n_heads, type=int)
    parser.add_argument('--transformer-encoder-n-layers', default=transformer_encoder_n_layers, type=int)
    parser.add_argument('--sil-spk-th', default=sil_spk_th, type=float, help='activity threshold to detect the silent speaker')
    parser.add_argument('--ahc-dis-th', default=ahc_dis_th, type=float, help='distance threshold above which clusters will not be merged')
    parser.add_argument('--clink-dis', default=clink_dis, type=float, help='modified distance corresponding to cannot-link')
    parser.add_argument('--threshold', default=threshold, type=str, help='DER thresholds')
    parser.add_argument('--median', default=median, type=str, help='DER median filter values')
    parser.add_argument('--collar', default=collar, type=float, help='the no-score collar (in +/- seconds) to attach to SPEAKER boundaries')
    
    args = parser.parse_args()
    return args
