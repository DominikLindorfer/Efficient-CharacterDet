# Efficient-CharacterDet

Are you tired using [tesseract](https://github.com/tesseract-ocr/tesseract) as your OCR engine? This repository provides a hackable [PyTorch](https://pytorch.org/) version of [EfficientDet](https://arxiv.org/abs/1911.09070) to detect characters, i.e. digits, letters and special symbols in numbers and words, within an image.
<p align="center">
  <img src="https://github.com/DominikLindorfer/BloombergReader/blob/main/bbg_screenshots/Promo.png" width="900">
</p>

### Datasets

For CharacterDet different datasets consisting of *512x512px* images with numbers, numbers with special characters and numbers, special characters and letters are given [here](./datasets/). Each dataset is created using the files [build_dataset.py](build_dataset.py) and [build_dataset_letters.py](/build_dataset_letters.py) where the latter is a more complex and advanced version of the former.

<p align="center">
  <img src="https://github.com/DominikLindorfer/BloombergReader/blob/main/bbg_screenshots/dataset_creation.png" width="700">
</p>

The total number of training/validation images is controlled using the parameters *total_ds_pics_train* and *total_ds_pics_train*. The size of the background canvas is controlled by 
*canvas_size=512* and can be set to a different value, if larger images are needed. Words and numbers are built using [images of individual characters](./bbg_numbers) which are resized using *img_size_x/y* and each character used to create the dataset is from the popular font [**Avenir LT Std Semi-Light**](https://fontsgeek.com/avenir-lt-std-font).

Please note that both the numbers and the letters datasets employ a horizontal and vertical shift of characters in their numbers/words but the character size is fixed in the former
while it is variable in the latter, as shown below.

<p align="center">
  <img src="https://github.com/DominikLindorfer/BloombergReader/blob/main/bbg_screenshots/Dataset_Difference.png" width="500">
</p>

This project uses the standard anchor scales/ratios as well as RGB mean and std used for by EfficientDet for COCO [given in the projects/*.yaml files](./projects). Obviously, these values do not reflect the datasets given here and a faster convergence or better loss could be found by optimizing these values. In my experiments (not shown in this repo) I did not achieve a better result playing with these values and for unclear reasons finding the anchor-ratios using [kmeans-anchros-ratios](https://github.com/mnslarcher/kmeans-anchors-ratios) failed to converge for me.

### EfficientDet Finetuning Procedure

I use the [COCO](https://cocodataset.org/) pretrained EfficientDet weights (pyTorch .pth) provided [here](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch/releases/tag/1.0) and finetune on the datasets provided above. I use a two-step training process, where in the first step the [EfficentNet backbone](https://arxiv.org/pdf/1905.11946.pdf) is frozen and only the head, i.e. the BiFPN layers as well as the 
Class/Box prediction net layers are trained. In the second step also the backbone layers are trainable. Provided below are the detailed training parameters for each CharacterDet downloadable from [here](/releases). All other parameters are chosen according to the [EfficientDet paper](https://arxiv.org/abs/1911.09070). Training is conducted on a single A100 40GB GPU.

TABLE

Tensorboard training-logs are provided [here](./logs). Please note that the snapshots provided above correspond to the respective step in these logs.

<p align="left">
  <img src="https://github.com/DominikLindorfer/BloombergReader/blob/main/video/Tensorboard_Loss_Ultra.png" width="650">
</p>

# References
This repository heavily relies on the ['Yet another EfficientDet in PyTorch' repo](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch).

