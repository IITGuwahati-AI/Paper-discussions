# Mask RCNN
Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick, 
Facebook AI Research (FAIR)

Mask RCNN is the paper that introduced instance segmentation using Faster RCNN and outperforms all existing, single-model entries on every task, including the
COCO 2016 challenge winners on tasks such as instance segmentation, boundingbox object detection, and person keypoint detection. It
efficiently detects objects in an image while simultaneously
generating a high-quality segmentation mask for each instance.


## Quick Overview
This paper provides a instance segmentation model which:
* Can generate high quality segmenation mask for each instance.
* Also efficiently detects bounding boxes.
* It performs good on person keypoint detection
* Extended Faster RCNN for giving mask.

## Summary
* Mask R-CNN extends Faster
R-CNN by adding a branch for predicting an object mask in
parallel with the existing branch for bounding box recognition.
* Mask R-CNN is simple to train and adds only a small
overhead to Faster R-CNN, running at 5 fps. Moreover,
Mask R-CNN is easy to generalize to other tasks, e.g., allowing us to estimate human poses in the same framework
* Proposed an RoIAlign layer that removes the harsh quantization of RoIPool, properly aligning
the extracted features with the input which improved results.
*  We can instantiate Mask R-CNN with multiple
architectures. For clarity, we can differentiate between: (i) the
convolutional backbone architecture used for feature extraction over an entire image, and (ii) the network head
for bounding-box recognition (classification and regression)
and mask prediction that is applied separately to each RoI.
* Using ResNeXt-101-FPN backbone compared to others, Mask R-CNN further improves results.
* Mask R-CNN decouples mask and class prediction: as the existing box
branch predicts the class label, it generate a mask for each
class without competition among classes (by a per-pixel sigmoid and a binary loss).

## Results
*  All instantiations of thier model outperform baseline variants of previous state-of-the-art models. This includes MNC
and FCIS , the winners of the COCO 2015 and 2016
segmentation challenges.
* Without bells and
whistles, Mask R-CNN with ResNet-101-FPN backbone
outperforms FCIS+++ , which includes multi-scale
train/test, horizontal flip test, and online hard example mining (OHEM).
* Mask RCNN performs well for Human Pose Estimation.

## Resources
* [Paper](https://arxiv.org/abs/1703.06870)
* [Video](https://www.youtube.com/watch?v=0vt05rQqk_I)

## Implementation
* The colab notebook with support with video can be found [here](https://colab.research.google.com/drive/1_qoMpkStoXKsdEiDpH6_wzH5neA49sct?usp=sharing).
* Implementation can be found [here](https://github.com/matterport/Mask_RCNN).
