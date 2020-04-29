# PeekBehind

PeekBehind for removing objects from photos from multiple angles on Python3, OpenCV, and 
[Mask R-CNN](https://github.com/matterport/Mask_RCNN).

![Work example](asserts/work_example.png)

The repository includes:
* Source code of PeekBehind built on Mask R-CNN and OpenCV.
* Training code for RTSD
* Pre-trained weights for RTSD
* Jupyter notebooks to visualize the detection and image blending
* Evaluation SSMI, PSNR, FID, L1 on Caltech Buildings dataset and Kaggle Architecture dataset

# Getting Started
* [peek_behind_demo](peek_behind_demo.ipynb) demonstration of the algorithm

* [detection_demo](detection_demo.ipynb) shows detection result on pre-trained weights

* ([coco_annotator.py](coco_annotator.py), 
[coco_utils](coco_utils.py), 
[detection](detection.py), 
[detection_train](detection_train.py)): These files contain definition of detection model

* [scores](scores.py), [choose_hyper](choose_hyper.py). Script for evaluation measures on test dataset and choosing hyper parameters

* [road_signs_dt_preprocessing](road_signs_dt_preprocessing.ipynb) This notebook converting an RTDS dataset for instance segmentation
# Custom mask example

![Custom mask example](asserts/custom_mask.png)

# Weights
- [RTSD dataset with segmentation masks](https://drive.google.com/open?id=1oSojJ2cB812GwWOobbAIFH003Qc18v_s)
- [COCO weight](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)
- [Pre-trained weight on RTSD](https://drive.google.com/open?id=1XvpmUh1ChGR7QNJmQuPe4gjJWuRpE6Dg)

# Evaluation

Evaluation SSMI, PSNR, FID, L1 on [dataset](https://drive.google.com/open?id=12_t1IyyY3E0pzQdbfkQF34lH5gnE9Irh)

Used datasets:
- [Architecture dataset](https://www.kaggle.com/wwymak/architecture-dataset#10_Mount_Clemens_Craftsman.jpg)
- [Caltech Buildings Dataset](http://www.mohamedaly.info/datasets/caltech-buildings)

### Methods
 - [Criminisi](https://github.com/igorcmoura/inpaint-object-remover/tree/master/inpainter)
 - [Deep image prior](https://github.com/DmitryUlyanov/deep-image-prior)
 - [Gated Convolution](https://github.com/JiahuiYu/generative_inpainting)

Results:

* Road signs

|Method    | SSMI  | FID  | L1  |PSNR |
|--------------------|-------|------|-----|-----|
|Criminisi           |  0.96 |  9.7 | 6.7% | 35.6|     
|Deep image prior    |  0.35 | 37.6 | 33.6%| 16.0|
|Gated Convolution          |  0.77 |  29.2| 12.0%| 25.3|
|Ours                | **0.98** | **5.6**  | **5.0%** | **38.3**|

* People

|Method    | SSMI  | FID  | L1  |PSNR |
|--------------------|-------|------|-----|-----|
|Criminisi           |  0.91 |  15.4 | 8.2% | 30.0|     
|Deep image prior    |  0.38 | 45.1 | 24.8%| 16.1|
|Gated Convolution          |  0.86 |  43.1| 11.3%| 25.8|
|Ours                | **0.98** | **3.9**  | **5.5%** | **34.8**|

* Cars

|Method    | SSMI  | FID  | L1  |PSNR |
|--------------------|-------|------|-----|-----|
|Criminisi           |  0.97 |  9.1 | 4.4% | 30.8|     
|Deep image prior    |  0.39 | 50.8 | 31.1%| 16.1|
|Gated Convolution          |   0.71 |  37.1| 12.4%| 22.2|
|Ours                | **0.98** | **5.6**  | **4.4%** | **33.2**|