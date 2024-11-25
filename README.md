# EEND-TBA


This repository provides the overall framework for training and evaluating End-to-End Neural Diarization framework with Token-based attractors proposed in **'EEND-TBA: End-To-End Neural Diarization Framework With Token-based Attractors'**

<img src=./EEND-TBA/figure/overall9.JPG>



## Result

you can see this framework performance in [score file](./EEND-TBA/score/0.scores)

also train and adapt avg .pt files are [train pt file](./EEND-TBA/modelpt/train/avg_trans.th) and [adapt pt file](./EEND-TBA/modelpt/adapt/avg_trans.th)

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



### Data Prepare


**데이터 세트 준비**
- Kaldi를 이용하여 화자분할을 위한 Simulated Data를 생성
  - [EEND repository](https://github.com/hitachi-speech/EEND)를 clone하여 Install tools 진행
  - [EEND/egs/callhome/v1/run_prepare_shared_eda.sh](https://github.com/hitachi-speech/EEND/blob/master/egs/callhome/v1/run_prepare_shared_eda.sh) 파일을 실행하여 data 생성
- [Preprocessing](https://github.com/Jungwoo4021/KT2023/tree/main/scripts/preprocessing_data/SpeakerDiarization)
  - simulated data 를 log mel spectrogram로 변환하여 .pickle(EEND_EDA) 또는 .npy(EEND_VC,EEND-TBA) 파일로 저장
  - 학습 코드에 preprocessing이 안되어있으면 진행하는 부분이 있기 때문에 별도로 진행하지 않아도 됨


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


### Folder Structure
```
EEND-TBA
│
├── config				: Config files for experiments
│   └── arguments.py	
├── Figures
│   ├── EERs.pdf
│   ├── 
│   ├── 
│   └── PartialSpoof_logo.png
├── modelpt				: Config files for experiments
│   ├── train			: convert string labels to numerical labels.
│   │   └── label2num_2cls_0sil		: bonafide/spoof (More to be released)
│   └── adapt			: convert string labels to numerical labels.
│       └── label2num_2cls_0sil		: bonafide/spoof (More to be released)
├── config				: Config files for experiments
│   └── arguments.py
├── src					: PartialSpoof Databases
│   ├── data						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   └── dev.lst		: waveform list
│   ├── infer						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   └── dev.lst		: waveform list
│   ├── log						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   └── dev.lst		: waveform list
│   ├── log						: Folder for dev set
│   │   ├── con_data	: related data file. (following kaldi format)
│   │   ├── con_wav		: waveform
│   │   └── dev.lst		: waveform list
│   └── vad
│       ├── dev
│       ├── eval
│       ├── eval
│       └── train
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
