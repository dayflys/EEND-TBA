import os
import datetime
import torch
import torch.nn as nn
import torch.utils.data as td
import numpy as np
from config.arguments import get_args
from src.log.controller import LogModuleController
from src.data.loaders import DiarizationDataLoader
from src.models.eend_vc import EEND_VC
from src.models.utils import prepare_model_for_eval
from src.train.scheduler import NoamScheduler
from src.train.train_handler import TrainHandler
from src.train.averagy import ModelAveragy
from src.train.loss import DiarizationLoss, SpeakerLoss
from src.infer.save_spkv import SaveSpkvLab
from src.infer.infer_handler import InferHandler
from src.data.features import get_input_dim
from src.data.preprocess import PreProcess


def set_experiment_environment(args):
    # reproducible
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['PYTORCH_SEED'] = str(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_ddp_environment(args):
    # DDP
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '4016'
    args.rank = args.process_id
    args.device = 'cuda:{}'.format(args.process_id)
    torch.cuda.empty_cache()
    torch.cuda.set_device(args.device)
    torch.distributed.init_process_group(
        backend='nccl', world_size=args.world_size, rank=args.rank)
    
    
def preprocess(args, loader):
    preprocess = PreProcess(args)
    preprocess_flag = f'{args.train_data_dir}/{args.preprocess_dir}/{args.preprocess_flag}'
    if not os.path.exists(preprocess_flag):
        dataset = loader.get_dataset_from_wav(args.train_data_dir)
        dataloader = loader.get_preprocess_dataloader(dataset)
        preprocess.run(dataloader, args.train_data_dir)
    preprocess_flag = f'{args.valid_data_dir}/{args.preprocess_dir}/{args.preprocess_flag}'
    if not os.path.exists(preprocess_flag):
        dataset = loader.get_dataset_from_wav(args.valid_data_dir)
        dataloader = loader.get_preprocess_dataloader(dataset)
        preprocess.run(dataloader, args.valid_data_dir)


def train(process_id, args):
    # experiment environment
    args.process_id = process_id
    args.flag_parent = process_id == 0
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    set_ddp_environment(args)

    # Setup logger
    if args.flag_parent:
        logger = LogModuleController.Builder(args.name, args.project
            ).tags([args.tags]
            ).save_source_files(args.path_scripts
            ).use_local(args.path_logging
            #).use_neptune(args.neptune_user, args.neptune_token
            ).use_wandb(args.wandb_group, args.wandb_entity, args.wandb_api_key
            ).build()
        logger.log_parameter(vars(args))
    else:
        logger = None

    # Prepare preprocess data
    loader = DiarizationDataLoader(args)
    preprocess(args, loader)

    # Prepare training data
    train_dataset = loader.get_dataset_from_feat(args.train_data_dir)
    train_sampler = td.DistributedSampler(train_dataset, shuffle=False) if not args.initmodel else None
    train_loader = loader.get_train_dataloader(train_dataset, train_sampler)

    val_dataset = loader.get_dataset_from_feat(args.valid_data_dir)
    val_sampler = td.DistributedSampler(val_dataset, shuffle=False) if not args.initmodel else None
    val_loader = loader.get_val_dataloader(val_dataset, val_sampler)

    all_num_speakers = train_dataset.get_allnspk()
    input_dim = loader.get_input_dim()
    
    # Set loss
    speaker_loss = SpeakerLoss(all_num_speakers, args.hidden_size)
    diarization_loss = DiarizationLoss()

    # Prepare model
    model = EEND_VC(args.num_speakers,
        input_dim = input_dim,
        hidden_size = args.hidden_size,
        transformer_encoder_n_heads = args.transformer_encoder_n_heads,
        transformer_encoder_n_layers = args.transformer_encoder_n_layers,
        transformer_encoder_dropout = args.transformer_encoder_dropout,
        shuffle=args.shuffle,
        speaker_loss = speaker_loss,
        diarization_loss = diarization_loss
        )
    
    device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
    model = model.to(device)
    if not args.initmodel: # for DDP 
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.device], find_unused_parameters=True)    
        torch.compile(model)    
    
    
        
    # Setup optimizer & scheduler
    if args.optimizer_name == 'noam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-9)
        scheduler = NoamScheduler(optimizer,
            args.hidden_size,
            warmup_steps=args.noam_warmup_steps,
            tot_step=len(train_loader),
            scale=1.0)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = None 
        #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(10, 50)), gamma=0.9) 
        

    # training
    train_handler = TrainHandler(
        args=args,
        model=model,
        loaders=(train_loader, val_loader),
        sampler=train_sampler,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger)

    #train_handler.save_train_model()
    train_handler.run()
    

def average(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices.split(',')[0]
    model_average = ModelAveragy(args)
    model_average.run()


def save_spkv_lab(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    loader = DiarizationDataLoader(args)
    dataset = loader.get_dataset_from_wav(args.data_dir)
    dataloader = loader.get_save_spkv_dataloader(dataset)
    input_dim = loader.get_input_dim()
    all_num_speakers = dataset.get_allnspk()

    # Set loss
    diarization_loss = DiarizationLoss()

    # Prepare model
    model = EEND_VC(
        num_speakers=args.num_speakers,
        input_dim=input_dim,
        hidden_size=args.hidden_size,
        transformer_encoder_n_heads=args.transformer_encoder_n_heads,
        transformer_encoder_n_layers=args.transformer_encoder_n_layers,
        transformer_encoder_dropout=0,
        shuffle=args.shuffle)
    model = prepare_model_for_eval(args.model_file, model)
    infer_handler = InferHandler(args, model)

    # save
    save_spkv = SaveSpkvLab(
        dataloader,
        infer_handler,
        diarization_loss,
        all_num_speakers=all_num_speakers,
        num_speakers=args.num_speakers,
        out_dir=args.out_dir)
    
    save_spkv.run()


def adapt(args):
    train(process_id=0, args=args)


def infer(args):
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    input_dim = get_input_dim(args.frame_size,args.context_size,args.input_transform)
    
    
    model = EEND_VC(args.num_speakers,
        input_dim = input_dim,
        hidden_size = args.hidden_size,
        transformer_encoder_n_heads = args.transformer_encoder_n_heads,
        transformer_encoder_n_layers = args.transformer_encoder_n_layers,
        transformer_encoder_dropout = 0,
        shuffle=args.shuffle
        )
    model = prepare_model_for_eval(args.model_file, model)
    
    
    
    list1 = [f'{i}' for i in range(2,7)] + ['all'] 
    original_data_dir = args.data_dir
    original_score_dir = args.score_dir
    
    for i in list1:
        
        args.data_dir = original_data_dir.format(i)
        args.score_dir = original_score_dir.format(f'/spk{i}')
        
        infer_handler = InferHandler(args, model)
        infer_handler.run()
        infer_handler.speakercount_scoreing()


def visual_sim(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    input_dim = get_input_dim(args.frame_size,args.context_size,args.input_transform)
    
    
    args.num_speakers = 3
    args.preprocess_dir = 'data'
    args.preprocess_trial = 'featlab_chunk_indices.txt'
    args.batchsize = 1
    args.num_workers = 4
    loader = DiarizationDataLoader(args)
    val_dataset = loader.get_dataset_from_feat(args.visual_sim_dir)
    val_loader = loader.get_val_dataloader(val_dataset, None)    
    iter_idx = 300

    model = EEND_VC(args.num_speakers,
        input_dim = input_dim,
        hidden_size = args.hidden_size,
        transformer_encoder_n_heads = args.transformer_encoder_n_heads,
        transformer_encoder_n_layers = args.transformer_encoder_n_layers,
        transformer_encoder_dropout = 0,
        shuffle=args.shuffle
        )
    model = prepare_model_for_eval(args.model_file, model)
    infer_handler = InferHandler(args, model)
    infer_handler.visual(val_loader,iter_idx,'sim')

def visual_ch(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    input_dim = get_input_dim(args.frame_size,args.context_size,args.input_transform)
    model = EEND_VC(args.num_speakers,
        input_dim = input_dim,
        hidden_size = args.hidden_size,
        transformer_encoder_n_heads = args.transformer_encoder_n_heads,
        transformer_encoder_n_layers = args.transformer_encoder_n_layers,
        transformer_encoder_dropout = 0,
        shuffle=args.shuffle
        )
    model = prepare_model_for_eval(args.model_file, model)
    args.num_speakers = 3
    args.preprocess_dir = 'data'
    args.preprocess_trial = 'featlab_chunk_indices.txt'
    args.batchsize = 1
    args.num_workers = 4
    loader = DiarizationDataLoader(args)
    val_dataset = loader.get_dataset_from_feat(args.visual_ch_dir)
    val_loader = loader.get_val_dataloader(val_dataset, None)
    iter_idx = 300
    infer_handler = InferHandler(args, model)
    infer_handler.visual(val_loader,iter_idx,'ch')
    

if __name__ == '__main__':
    (train_args, save_spkv_lab_args, adapt_args, infer_args) = get_args()
    set_experiment_environment(train_args)
    # start
    '''
    torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.spawn(
        train, 
        nprocs=train_args.world_size, 
        args=(train_args,)
    )
    average(train_args)
    save_spkv_lab(save_spkv_lab_args)
    '''
    
    # adapt(adapt_args)
    
    # average(adapt_args)
    # infer(infer_args)
    visual_sim(infer_args)
    visual_ch(infer_args)
    #infer_handler = InferHandler(infer_args)
    #infer_handler.scoring()
