[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-video-denoising/video-denoising-on-set8-sigma30)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma30?p=unsupervised-deep-video-denoising)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-video-denoising/video-denoising-on-set8-sigma40)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma40?p=unsupervised-deep-video-denoising)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unsupervised-deep-video-denoising/video-denoising-on-set8-sigma50)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma50?p=unsupervised-deep-video-denoising)

# Unsupervised Deep Video Denoising 

To appear at **IEEE/CVF International Conference on Computer Vision (ICCV), 2021**.

Authors: **Dev Yashpal Sheth\*, Sreyas Mohan\*, Joshua Vincent, Ramon Manzorro, Peter A. Crozier, Mitesh M. Khapra, Eero P. Simoncelli and Carlos Fernandez-Granda** [\* - Equal Contribution].

Paper: [arXiv:2011.15045](https://arxiv.org/abs/2011.15045)

Website: [https://sreyas-mohan.github.io/udvd/](https://sreyas-mohan.github.io/udvd/)

## Pre-trained Models

The `pretrained` folder contains the saved models, details about each are listed below.
1. `blind_video_net.pt` - UDVD trained on the *DAVIS* dataset and Gaussian noise with *sigma = 30*.
2. `blind_spot_net.pt` - UDVD (1 frame) which is simply a unsupervised deep image denoiser.
3. `fast_dvd_net.pth` - Pretrained FastDVDnet model taken directly from [https://github.com/m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet).
4. `fluoro_micro.pt` - UDVD trained on the Fluorescence Microscopy dataset.
5. `raw_video.pt` - UDVD trained on the test set of the Raw Video dataset.
6. `single_video_Set8_rafting_30.pt` - UDVD-S trained on a single noisy video sequence *rafting* from the *GoPro* set with Gaussian noise *sigma = 30*. Similarly pretrained models for the other 3 seqeunces in the *GoPro* set have also been released i.e. *hypersmooth, motorbike, snowboard*. 
7. `mf2f_online_with_teacher_rafting_30.pth` - MF2F model which is a fine-tuned FastDVDnet directly on the test sequence *rafting* from the *GoPro* set with Gaussian noise *sigma = 30*. We used the official implementation at [https://github.com/centreborelli/mf2f](https://github.com/centreborelli/mf2f).

## Jupyter Notebook Demos

We provide the following demos in the `notebook_demos` folder.
1. `denoising_demo.ipynb` - Basic usage of pretrained models on courrupted videos.
2. `evaluation_demo.ipynb` - Evaluation of UDVD on the *Set8* dataset and UDVD-S on *rafting* from the *GoPro* set with Gaussian noise *sigma = 30*.
3. `analysis_demo.ipynb` - Video denoising as spatiotemporal adaptive filtering and implicit motion compensation.
4. `microscopy_demo.ipynb` - UDVD demo on the Fluorescence Microscopy dataset.
5. `raw_video_demo.ipynb` - UDVD demo on the Raw Video dataset.

## Datasets

We use the following datasets as part of our paper. Download links to each has been listed below. Note that the *Set8* dataset consists of 4 sequences of the *GoPro* set and 4 sequences of the *Derfs* set. Please refer to the supplementary material in the paper for details on the exact sequences used.
1. `DAVIS` - Primairy dataset on which the natural videos model was trained. [https://davischallenge.org/davis2017/code.html](https://davischallenge.org/davis2017/code.html)
2. `GoPro` - Released with the FastDVDnet paper. [https://github.com/m-tassano/fastdvdnet](https://github.com/m-tassano/fastdvdnet)
3. `Derfs` - Contains 4 sequences of the *Set8* set and 3 more were used to compare with MF2F. [https://media.xiph.org/video/derf/](https://media.xiph.org/video/derf/)
4. `Vid3oC` - Part of the AIM 2020 Video Extreme Super-Resolution Challenge. [https://competitions.codalab.org/competitions/24685](https://competitions.codalab.org/competitions/24685)
5. `CTC` - Fluorescence Microscopy dataset. [http://celltrackingchallenge.net/2d-datasets/](http://celltrackingchallenge.net/2d-datasets/)
6. `RawVideo` - Released as part of the RViDeNet paper. [https://github.com/cao-cong/RViDeNet](https://github.com/cao-cong/RViDeNet) 

## Training

To train UDVD on the *DAVIS* dataset.
```shell
python train.py \
        --model blind-video-net-4
        --data-path dataset/DAVIS
        --dataset DAVIS
        --batch-size 32
        --lr 1e-4
        --num-epochs 40
```

To train UDVD-S on the *rafting* sequence from the *GoPro* set with Gaussian noise *sigma = 30*.
```shell
python single_train.py \
        --model blind-video-net-4
        --data-path dataset/Set8
        --dataset SingleVideo
        --dataset-aux GoPro
        --video rafting
        --aug 2
        --sample
        --heldout
        --batch-size 8
        --lr 1e-4
        --num-epochs 32
        --step-checkpoints
```

To train UDVD on the Fluorescence Microscopy dataset.
```shell
python fluoro_train.py \
        --model blind-video-net-4
        --channels 1
        --out-channels 1
        --loss mse
        --data-path datasets/CTC
        --dataset CTC
        --batch-size 32
        --lr 1e-4
        --num-epochs 40
        --step-checkpoints
```

To train UDVD on the Raw Video dataset.
```shell
python raw_train.py \
        --model blind-video-net-4
        --channels 1
        --out-channels 1
        --loss mse
        --data-path datasets/RawVideo
        --dataset RawVideo
        --batch-size 8
        --lr 1e-4
        --num-epochs 4
        --step-checkpoints
```
## Citation

```
@InProceedings{Sheth_2021_ICCV,
    author = {Sheth, Dev Yashpal and Mohan, Sreyas and Vincent, Joshua and Manzorro, Ramon and Crozier, Peter A. and Khapra, Mitesh M. and Simoncelli, Eero P. and Fernandez-Granda, Carlos},
    title = {Unsupervised Deep Video Denoising},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2021}
}
```
