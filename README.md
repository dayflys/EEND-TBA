# EEND-TBA


This repository provides the overall framework for training and evaluating End-to-End Neural Diarization framework with Token-based attractors proposed in **'EEND-TBA: End-To-End Neural Diarization Framework With Token-based Attractors'**

<img src=./EEND-TBA/figure/overall_structure.JPG>



## Result

you can see this framework performance in [score file](./EEND-TBA/score/0.scores)

Also, train and adapt avg .pt files are [train pt file](./EEND-TBA/modelpt/train/avg_trans.th) and [adapt pt file](./EEND-TBA/modelpt/adapt/avg_trans.th)

* EEND-TBA
    * DataSet 

        - train: 3spk Simulated Dataset
        
        - adapt : Callhome1 Dataset
        
        - test : Callhome2 Dataset

    1. DER (%)
         * number of speakers: 2, 3, 4, 5, 6, ALL
         * DER: 6.68, 11.37, 14.68, 20.22, 28.03, 11.64
        
    2. Speaker Counting (%)
        
        |- |2  |3 |4 |5 |6 ||ref|pred|
        |--|-- |--|--|--|--|--|--|--|
        |2 |**134**|16 |2 |0 |0 ||148 |152 |
        |3 |14 |**58** |8 |0 |0 ||74 |80 |
        |4 |0  |0 |**8** |2 |0 ||20 |10 |
        |5 |0  |0 |2 |**2** |1 ||5 |5 |
        |6 |0  |0 |0 |0 |**1** ||3 |1 |
        |7+|0  |0 |0 |1 |1 ||0 |2 |


---

## Run experiment

### Set system arguments

First, you need to set system arguments. You can set arguments in `config/arguments.py`.


### Experiment Setup

Our environment (for GPU training)

Based on a below docker image  

GPU: 4 NVIDIA A5000

<a href="https://github.com/Jungwoo4021/KT2023/blob/main/scripts/docker_files/Dockerfile24_09"><img src="https://img.shields.io/badge/DOCKER FILE-2496ED?style=for-the-badge&logo=Docker&logoColor=white"></a>


```
Docker file summary

Docker
    nvcr.io/nvidia/pytorch:23.08-py3 

Python
    3.8.12

Pytorch 
    2.1.0a0+29c30b1

Torchaudio 
    2.0.1
```



---

### Data Prepare


#### first, Preparing Simulated Data using kaldi 

1. git clone [EEND repository](https://github.com/hitachi-speech/EEND) 
2. Following EEND ropository, Install necessary tools
3. Run [EEND/egs/callhome/v1/run_prepare_shared_eda.sh](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared_eda.sh) file to make Simulated data



#### Second, Data Preprocessing 

- Convert the simulated data to log mel spectrogram format and save it as .npy file 

- If the preprocessing step is not included in the training code, there might be a section to handle this automatically, so you do not need to perform it separately


---

### Additional logger

We have a basic logger that stores information in local. However, if you would like to use an additional online logger (wandb or neptune):

```In arguments.py
# Wandb: Add 'wandb_user' and 'wandb_token'
# Neptune: Add 'neptune_user' and 'neptune_token' 
```

#### for example

```
dictionary:
'wandb_group'   : 'group',
'wandb_entity'  : 'user-name',
'wandb_api_key' : 'WANDB_TOKEN',
'neptune_user'  : 'user-name',
'neptune_token' : 'NEPTUNE_TOKEN'
```


#### logger
```
In main.py

# Just remove "#" in logger
builder = egg_exp.log.LoggerList.Builder(args['name'], args['project'], args['tags'], 	
                                         args['description'], args['path_scripts'], args)
builder.use_local_logger(args['path_log'])
# builder.use_neptune_logger(args['neptune_user'], args['neptune_token'])
# builder.use_wandb_logger(args['wandb_entity'], args['wandb_api_key'], 
# 												 args['wandb_group'])
logger = builder.build()
logger.log_arguments(experiment_args)
```



---
### Multi-GPU training


GPU indices could be set before training using the command export CUDA_VISIBLE_DEVICES=0,1,2,3.

default value is CUDA_VISIBLE_DEVICES=0,1

### Training
1. Single GPU
```
python main.py 
```
2. Multiple GPU
```
python main.py --cuda_visible_devices=0,1
```

### Inference
1. Single GPU (only)


---


### Folder Structure
```
EEND-TBA
│
├── config				: Config files for experiments
│   └── arguments.py	
├── Figures
│   ├── EEND_TBA_74_batch_img.png
│   ├── EEND_VC_74_batch_img.png
│   ├── EEND_TBA_126_batch_img.png
│   ├── EEND_VC_126_batch_img.png    
│   └── overall_structure.JPG       : overall structure of EEND-TBA 
├── modelpt				            : averagy weights of best performance in each phase 
│   ├── train			
│   │   └── avg_trans.th
│   └── adapt			
│       └── avg_trans.th
├── score				            : Score file of best performance 
│   └── 0.scores
├── src					
│   ├── data						: Folder for data
│   │   ├── datasets.py	            
│   │   ├── features.py	            
│   │   ├── kaldi.py	            
│   │   ├── loader.py		        
│   │   └── preprocess.py	        
│   ├── infer						: Folder for inference 
│   │   ├── infer_handler.py	            
│   │   └── save_spkv.py		            
│   ├── log						    : logger 
│   │   ├── controller.py	            
│   │   ├── interface.py		        
│   │   ├── local.py	            
│   │   ├── neptune.py		         
│   │   └── wandb.py		         
│   ├── models						: Folder for EEND-TBA model
│   │   ├── eend_tba.py	            
│   │   ├── transformer.py		    
│   │   └── utils.py	            
│   └── train
│       ├── averagy.py
│       ├── loss.py
│       ├── scheduler.py
│       └── train_handler.py
├── docker_build.sh
├── docker_run.sh
├── Dockerfile          
└── main.py
```





<!-- Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset [4]. -->

<!-- python ./download_dataset.py
(Alternative) Manual preparation is available via

ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
Download LA.zip and unzip it
Set your dataset directory in the configuration file
Training
The main.py includes train/validation/evaluation. -->
