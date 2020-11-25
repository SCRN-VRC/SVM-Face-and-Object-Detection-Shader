<img src="/Images/Project.png" align="middle" width="3000"/>

# Face and Object Detection Shader

A binary classifier implemented with a support vector machine (SVM) written in C++ and converted to HLSL to be used inside VRC. I have included a program that allows you to train your own detector. You can use it to detect other objects as well.

## Overview
<img src="/Images/SVM.png" align="middle" width="3000"/>

My implementation has three camera inputs to detect varying sizes 480 x 270, 240 x 135, 120 x 67. A 64 x 64 sliding window is applied with a stride of 16 pixels for the largest camera inputs to 8 pixels for the smallest. HOG features are extracted by binning by the magnitude of the gradient according to the direction of the gradient into 8 bins. The features are normalized to account for differences in lighting. The radial basis function (RBF) kernel is applied to each of the 1568 features per 64 x 64 image block and the support vectors. At the end, classification is done by doing the summation of all RBF calculations per 64 x 64 image block.

## Problems

I have fixed most of the problems I listed in the previous implementation. However,
1. Cameras are friends only, you can only show this to your friends.
2. SVMs needs to store data of the hyperplane that does the classification. So more training = more complex hyper plane = lag in game.

## Unity and VRChat Setup

## Contact

Contact me on Discord if you have any questions or suggestions: **SCRN#8008**

Thanks to [Merlin](https://github.com/Merlin-san) and 1001 for making this possible.
