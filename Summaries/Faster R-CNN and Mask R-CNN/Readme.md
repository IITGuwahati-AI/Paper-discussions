# Faster R-CNN and Mask R-CNN

## Faster R-CNN 
   *Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun*
 - Faster R-CNN builds upon the same idea of detection network as Fast R-CNN but the region proposal network shares     the computation with the detection network here.
 - The use of novel pyramid of anchors is the key factors which enables the sharing of convolutional layers between the two networks.
 -Alternate Training is used for the unified network of Region Proposal Network and Fast R-CNN detection network.
 
 *The slides used in the pressentation of the paper can be found in the folder with Faster_RCNN.pdf name and the paper used with Faster_RCNN_Paper.pdf*
 
 ## Mask R-CNN
 *Kaiming He, Georgia Gkioxari, Piotr Dollar, Ross Girshick,*
 *Facebook AI Research (FAIR)*
 
 - Mask R-CNN build upon the network of Faster R-CNN with a small overhead to its architecture i.e. the mask branch   for parallel instance segmentation on RoI.
 - The use of RoIAlign instead of RoIPool gives it boost in precision by about 50% and is one of key steps.
 - Decoupling of mask and class prediction is the other key steps which reduces competion among masks of other classes.
 - The code for Mask R-CNN can be found at :
      - [Code](https://github.com/matterport/Mask_RCNN)
  
 *The slides use in the presentation for the paper can be found in the folder under the name Mask_RCNN.pdf and the paper used with Mask_RCNN_Paper.pdf*
 
 
