# QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection

This repository contains the code and pretrained models for the following paper:

* **Title** : QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection
* **Autor** : Duc-Tuan Truong, Tianchi Liu, Ruijie Tao, Junjie Li, Kong Aik Lee, Eng Siong Chng

## Pretrained Model
The pretrained model XLSR can be found at [link](https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt).

We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/truongdu001_e_ntu_edu_sg/EhIOVqPCD69CpJ9qFsm7qhcB3UFN2hGkif4Mfr8OzkwURQ?e=asrfwL). 

## Setting up environment
Python version: 3.7.16

Pip version: 22.3.1

Install PyTorch
```bash
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

Install other libraries:
```bash
pip install -r requirements.txt
```

Install fairseq:
```bash
git clone https://github.com/facebookresearch/fairseq.git fairseq_dir
cd fairseq_dir
git checkout a54021305d6b3c
pip install --editable ./
```

## Training & Testing
Before run training, replace 2019LA train metadata by the meatadata with MOS in `database/protocols_w_mos/ASVspoof_LA_cm_protocols` 

To train and produce the score for LA set evaluation, run:
```bash
python main.py --model conformer_tcm
```

## Scoring
To get evaluation results of minimum t-DCF and EER (Equal Error Rate), follow these steps:

For ITW & FoR track evaluation
```bash
python score_ood_dataset.py your_ITW_score.txt database/ood_ds_keys/ITW/
python score_ood_dataset.py your_FoR_score.txt database/ood_ds_keys/FoR/
```
For LA & DF track evaluation
```bash
cd 2021/eval-package
python main.py --cm-score-file your_LA_score.txt --track LA --subset eval 
python main.py --cm-score-file your_DF_score.txt --track DF --subset eval
```
<!-- ## Inference
To run inference on a single wav file with the pretrained model, run:
```bash
python inference.py --ckpt_path=path_to/model.pth --threshold=-3.73 --wav_path=path_to/audio.flac
```
The threshold can be obtained when calculating EER on one of the evaluation sets. In this example, the threshold is from DF set evaluation. -->

## Citation
If you find our repository valuable for your work, please consider giving a star to this repo and citing our paper:
```
@misc{truong2025qamoqualityawaremulticentroidoneclass,
      title={QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection}, 
      author={Duc-Tuan Truong and Tianchi Liu and Ruijie Tao and Junjie Li and Kong Aik Lee and Eng Siong Chng},
      year={2025},
      eprint={2509.20679},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.20679}, 
}
```

### Acknowledge
We use some parts of the following codebases:

[AIR-ASVspoof](https://github.com/yzyouzhang/AIR-ASVspoof) (for One-class learning loss).

[scoreq](https://github.com/alessandroragano/scoreq) (for obtaining MOS).

Thanks for these authors for sharing their work!
