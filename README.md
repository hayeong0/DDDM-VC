## DDDM-VC: Decoupled Denoising Diffusion Models with Disentangled Representation and Prior Mixup for Verified Robust Voice Conversion

The official Pytorch implementation of DDDM-VC (AAAI 2024)

[Ha-Yeong Choi*](https://github.com/hayeong0), [Sang-Hoon Lee*](https://github.com/sh-lee-prml), Seong-Whan Lee 

## [Paper](https://arxiv.org/abs/2305.15816) | [Project Page](https://hayeong0.github.io/DDDM-VC-demo/) | [Audio Sample](https://dddm-vc.github.io/demo/)

![image](https://github.com/hayeong0/DDDM-VC/assets/47182864/8c2e862a-5ac2-4720-b8fd-0d8967bcc92b)
<p align="center"><em> Overall architecture </em>

> Diffusion-based generative models have recently exhibited powerful generative performance. However, as many attributes exist in the data distribution and owing to several limitations of sharing the model parameters across all levels of the generation process, it remains challenging to control specific styles for each attribute. To address the above problem, we introduce decoupled denoising diffusion models (DDDMs) with disentangled representations, which can enable effective style transfers for each attribute in generative models. In particular, we apply DDDMs for voice conversion (VC) tasks, tackling the intricate challenge of disentangling and individually transferring each speech attributes such as linguistic information, intonation, and timbre. First, we use a self-supervised representation to disentangle the speech representation. Subsequently, the DDDMs are applied to resynthesize the speech from the disentangled representations for style transfer with respect to each attribute. Moreover, we also propose the prior mixup for robust voice style transfer, which uses the converted representation of the mixed style as a prior distribution for the diffusion models. The experimental results reveal that our method outperforms publicly available VC models. Furthermore, we show that our method provides robust generative performance even when using a smaller model size.


## 📑 Pre-trained Model
Our model checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1tDIQ5Nv-X2svhcww35LWMC1El3SDlI_I?usp=sharing).

- model_base.pth
- voc_ckpt.pth
- voc_bigvgan.pth
- f0_vqvae.pth



## ⚙️ Setup
1. Clone this rep && Install python requirement

```
git clone https://github.com/hayeong0/DDDM-VC.git
pip install -r req*
``` 
2. Download the pre-trained model checkpoint from drive.

## 🔨 Usage
### Preprocess
1. Data
- Training requires both wav files and F0 features, which we extract using YAAPT through the script `./preprocess/extract_f0.py`.
- After extracting F0, create a list of files with the path to each data item, as shown in the following example:
```
train_wav.txt
/workspace/raid/dataset/LibriTTS_16k/train-clean-360/100/121669/100_121669_000001_000000.wav
/workspace/raid/dataset/LibriTTS_16k/train-clean-360/100/121669/100_121669_000003_000000.wav

train_f0.txt
/workspace/raid/dataset/LibriTTS_f0_norm/train-clean-360/100/121669/100_121669_000001_000000.pt
/workspace/raid/dataset/LibriTTS_f0_norm/train-clean-360/100/121669/100_121669_000003_000000.pt
```

2. F0_VQVAE
- We trained the f0_vqvae model using [SpeechResynthesis repository](https://github.com/facebookresearch/speech-resynthesis).


### 🔍 Training
- For training, prepare a file list with the following structure:
```
|-- filelist 
|    |-- train_f0.txt
|    |-- train_wav.txt
|    |-- test_f0.txt
|    `-- test_wav.txt
```
- Run `train_dddmvc.py`


### 🔑 Inference
- Run `infer.sh`
 
```
bash infer.sh

python3 inference.py \
    --src_path './sample/src_p227_013.wav' \
    --trg_path './sample/tar_p229_005.wav' \
    --ckpt_model './ckpt/model_base.pth' \
    --ckpt_voc './vocoder/voc_ckpt.pth' \
    --ckpt_f0_vqvae './f0_vqvae/f0_vqvae.pth' \
    --output_dir './converted' \
    -t 6
```

🎧 Train and test it on your own dataset and share your interesting results! 🤗



## 🎓 Citation
```
@inproceedings{choi2024dddm,
  title={Dddm-vc: Decoupled denoising diffusion models with disentangled representation and prior mixup for verified robust voice conversion},
  author={Choi, Ha-Yeong and Lee, Sang-Hoon and Lee, Seong-Whan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={16},
  pages={17862--17870},
  year={2024}
}
```



## 💎 Acknowledgements
- [DiffVC](https://github.com/huawei-noah/Speech-Backbones/tree/main/DiffVC): for overall diffusion src code
- [Speech-Resynthesis](https://github.com/facebookresearch/speech-resynthesis): for f0-vqvae
- [HiFiGAN](https://github.com/jik876/hifi-gan): for vocoder
- [torch-nansypp](https://github.com/revsic/torch-nansypp): for data augmentation

## License
This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg










