# [CVPR2024] SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting
## [Paper]() | [Video Youtube](https://youtu.be/IzC-fLvdntA) | [Project Page](https://initialneil.github.io/SplattingAvatar)

Official Repository for CVPR 2024 paper [*SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting*](https://cvpr.thecvf.com/Conferences/2024/AcceptedPapers). 

<img src="assets/SplattingAvatar-demo.gif" width="800"/> 


<!-- - Overview -->
<img src="assets/SplattingAvatar-teaser.jpg" width="800"/> 
<!-- - Framework -->
<img src="assets/SplattingAvatar-framework.jpg" width="800"/> 

### Lifted optimization
The embedding points of 3DGS on the triangle mesh are updated by the *walking on triangle* scheme.
See the `phongsurface` module implemented in c++ and pybind11.
<img src="assets/SplattingAvatar-triangle.jpg" width="800"/> 

## Getting Started
- Clone [the official 3DGS](https://github.com/graphdeco-inria/gaussian-splatting).
```
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```
- Clone this repo.
```
git clone https://github.com/initialneil/SplattingAvatar.git
```
- Setup conda env.
```
```

- Download [FLAME model](https://flame.is.tue.mpg.de/download.php), choose **FLAME 2020** and unzip it, copy 'generic_model.pkl' into `./model/imavatar/FLAME2020`


## Preparing dataset
We provide the preprocessed data of the 10 subjects used in the paper.
- Our preprocessing followed [IMavatar](https://github.com/zhengyuf/IMavatar/tree/main/preprocess#preprocess) and replaced the *Segmentation* with [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting).
- Pre-trained checkpoints are provided together with the data.
- [Google Drive](https://drive.google.com/drive/folders/1YPEG1IYgkZWTlibRZfMjhXFMw58JJVeq?usp=sharing) or [Baidu Disk](https://pan.baidu.com/s/12ycpotyE4KUZ-HvhpCcVxw?pwd=bkfh)

<img src="assets/SplattingAvatar-dataset.jpg" width="800"/> 



## Training
```
python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir <path/to/subject>
# for example:
python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir C:/SplattingAvatar/bala

# use SIBR_remoteGaussian_app.exe from 3DGS to watch the training
SIBR_remoteGaussian_app.exe --path <path/to/model_path>
# for example:
SIBR_remoteGaussian_app.exe --path C:\SplattingAvatar\bala\output-splatting\last_checkpoint

# it is recommended to change "FPS" to "Trackball" in the viewer
# you don't need to chagne the "path" everytime
```
## Evaluation
```
python eval_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir <path/to/model_path>
# for example:
python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir C:/SplattingAvatar/bala/output-splatting/last_checkpoint
```


## GPU requirement
We conduct our experiments on a single NVIDIA RTX 3090 with 24GB.
Training with less GPU memory can be achieved by setting a maximum number of Gaussians.
```
# in configs/splatting_avatar.yaml
model:
  max_n_gauss: 300000 # or less as needed
```
or set by command line
```
python train_splatting_avatar.py --config configs/splatting_avatar.yaml --dat_dir <path/to/subject> model.max_n_gauss=300000
```

## Citation
If you find our code or paper useful, please cite as:
```
@inproceedings{SplattingAvatar:CVPR2024,
  title = {{SplattingAvatar: Realistic Real-Time Human Avatars with Mesh-Embedded Gaussian Splatting}},
  author = {Shao, Zhijing and Wang, Zhaolong and Li, Zhuang and Wang, Duotun and Lin, Xiangru and Zhang, Yu and Fan, Mingming and Wang, Zeyu},
  booktitle = {Computer Vision and Pattern Recognition (CVPR)},
  year = {2024}
}
```

## Acknowledgement
We thanks the following authors for their excellent works!
- [instant-nsr-pl](https://github.com/bennyguo/instant-nsr-pl)
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [IMavatar](https://github.com/zhengyuf/IMavatar)
- [INSTA](https://github.com/Zielon/INSTA)


