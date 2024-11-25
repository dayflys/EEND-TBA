# EEND-TBA

This repository provides the overall framework for training and evaluating audio anti-spoofing systems proposed in 'AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks'

## Getting started
requirements.txt must be installed for execution. We state our experiment environment for those who prefer to simulate as similar as possible.

Installing dependencies
pip install -r requirements.txt
Our environment (for GPU training)
Based on a docker image: pytorch:1.6.0-cuda10.1-cudnn7-runtime
GPU: 1 NVIDIA Tesla V100
About 16GB is required to train AASIST using a batch size of 24
gpu-driver: 418.67
Data preparation
We train/validate/evaluate AASIST using the ASVspoof 2019 logical access dataset [4].

python ./download_dataset.py
(Alternative) Manual preparation is available via

ASVspoof2019 dataset: https://datashare.ed.ac.uk/handle/10283/3336
Download LA.zip and unzip it
Set your dataset directory in the configuration file
Training
The main.py includes train/validation/evaluation.

To train AASIST [1]:

python main.py --config ./config/AASIST.conf
To train AASIST-L [1]:

python main.py --config ./config/AASIST-L.conf
Training baselines
We additionally enabled the training of RawNet2[2] and RawGAT-ST[3].

To Train RawNet2 [2]:

python main.py --config ./config/RawNet2_baseline.conf
To train RawGAT-ST [3]:

python main.py --config ./config/RawGATST_baseline.conf

















---


**Result**

* EEND-TBA
    * Test Data: CALLHOME Data 
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

             
    * 훈련 데이터 세트: Simulated Dataset (train) / Callhome1 Dataset (adapt)/ Callhome2 Dataset (test)
    

# Run experiment


### Set system arguments

First, you need to set system arguments. You can set arguments in `config/arguments.py`.


### Experiment Setup

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


### Data


**데이터 세트 준비**
- Kaldi를 이용하여 화자분할을 위한 Simulated Data를 생성
  - [EEND repository](https://github.com/hitachi-speech/EEND)를 clone하여 Install tools 진행
  - [EEND/egs/callhome/v1/run_prepare_shared_eda.sh](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared_eda.sh) 파일을 실행하여 data 생성
- [Preprocessing](https://github.com/Jungwoo4021/KT2023/tree/main/scripts/preprocessing_data/SpeakerDiarization)
  - simulated data 를 log mel spectrogram로 변환하여 .pickle(EEND_EDA) 또는 .npy(EEND_VC,EEND-TBA) 파일로 저장
  - 학습 코드에 preprocessing이 안되어있으면 진행하는 부분이 있기 때문에 별도로 진행하지 않아도 됨

<!-- 
The VoxCeleb datasets are used for these experiments.

The train list should contain the identity and the file path, one line per utterance, as follows:

```
id00000 id00000/youtube_key/12345.wav
id00012 id00012/21Uxsk56VDQ/00001.wav
``` -->

### Multi-GPU training


GPU indices could be set before training using the command export CUDA_VISIBLE_DEVICES=0,1,2,3.

default value is CUDA_VISIBLE_DEVICES=0,1

If you are running more than one distributed training session, you need to change the --port argument.


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

```
PartialSpoof
├── 01_download_database.sh			: Script used to download PartialSpoof from zenodo.
├── 03multireso
│   ├── 01_download_pretrained_models.sh	: Script used to download pretrained models.
│   ├── main.py
│   ├── model.py			: Model structure and loss are in here! same for multi/single-reso.
│   ├── multi-reso		: folder for multi-reso model
│   ├── README.md
│   └── single-reso		: folder for single-reso model
│       └── {2, 4, 8, 16, 32, 64, utt}
├── config_ps				: Config files for experiments
│   ├── config_test_on_dev.py
│   └── config_test_on_eval.py
├── env.sh						
├── Figures
│   ├── EERs.pdf
│   └── PartialSpoof_logo.png
├── LICENSE
├── metric			
│   ├── cal_EER.sh
│   ├── RangeEER.py
│   ├── README.md
│   ├── rttm_tool.py
│   ├── SegmentEER.py
│   └── UtteranceEER.py
├── database					: PartialSpoof Databases
│   ├── train
│   ├── dev						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   └── dev.lst		: waveform list
│   ├── eval
│   ├── label2num			: convert string labels to numerical labels.
│   │   └── label2num_2cls_0sil		: bonafide/spoof (More to be released)
│   ├── protocols
│   ├── segment_labels
│   └── vad
│       ├── dev
│       ├── eval
│       └── train
├── modules
│   ├── gmlp.py
│   ├── LICENSE
│   ├── multi_scale
│   │   └── post.py
│   ├── s3prl  	     			: s3prl repo 
│   └── ssl_pretrain 			: Folder to save downloaded pretrained ssl model
├── project-NN-Pytorch-scripts.202102	: Modified project-NN-Pytorch-scripts repo
└── README.md
```