# C2Net-Pytorch Code backup

## Introduction

Code for the C2Net complemented with Pytorch. 

## Environment

Please use the anaconda 3.7 and run: 

```
conda create --name c2net --file specf_c2net.txt
```


## Data Preparation

The Data Preparations are following Guoxia Wang with his [DOOBNet](https://github.com/GuoxiaWang/DOOBNet). You should set the root path of the dataset to ```data/PIOD```, or set the dataset path in  ```experiments/configsPIOD_myUnet_kxt.yaml```.


## Training

For training C2Net on PIOD training dataset, you can run:

```
cd $ROOT/detect_occ/
python train_val_lr.py --config ../experiments/configs/PIOD_myUnet_kxt.yaml --gpus 0
```

You can also download the [pretrained model](https://pan.baidu.com/s/12pO07b3gSNMI_E5GZagmGw) (baiduyun code:f7f8) of PIOD and put it in  ```experiments/output/```. Then you can evaluate the results by run: 
```
python train_val_lr.py --config ../experiments/configs/PIOD_myUnet_kxt.yaml --evaluate --resume 2021-01-05_02-47-44/checkpoint_19.pth.tar --gpus 1
```


## Evaluation

The output files of testset are in ```.mat``` format (same as DOOBNet) and save in ```experiments/output/``` . The Evaluations are following Guoxia Wang with his [DOOBNet](https://github.com/GuoxiaWang/DOOBNet). 

The evaluation results in PIOD: 

|  Method   |   ODS-E   |   OIS-E   |   AP-E   |   ODS-O   |   OIS-O   |   AP-O   |
| ---- | --- | --- | --- | --- | --- | --- |
| SRF-OCC | .345 | .369 | .207 | .268 | .286 | .152 |
| DOC-HED  | .509 | .532| .468 | .460 | .479 | .405 |
| DOC-DMLFOV | .669 | .684 | .677 | .601 | .611 | .585 |
| DOOBNet | .736 | .746 | .723 | .702 | .712 | .683 |
| OFNet | .751 | .762 | .773 | .718 | .728 | .729 |
| Ours | .777 | .783 | .820 | .730 | .732 | .741 |
