[<img src="https://i.imgur.com/G1Pnf8X.jpg" align="middle" width="3000"/>](https://streamable.com/8q355)

# Face and Object Detection for VRChat

Implemented with a support vector machine (SVM) written in C++ and converted to HLSL to be used inside VRC on avatars. I have included a program that allows you to train your own detector and use it in game. Not only can you use it to detect faces, but other objects as well. **It's not very accurate**, I'll cover this in a later section below.

## Overview
<img src="SVM.png" align="middle" width="3000"/>

A 160x90 camera input is convereted into black and white with the gradient extracted. A 64x64 sliding window is applied with a stride of 4 pixels. HOG features are extracted by binning by the magnitude of the gradient according to the direction of the gradient into 8 bins. The features are normalized to account for differences in lighting and stretched out into a bigger texture to reduce the number of conditional moves and texture reads. The radial basis function (RBF) kernel is applied to each of the 1568 features per 64x64 image block and the support vectors. At the end, classification is done by doing the summation of all RBF calculations per 64x64 image block.

## Problems

1. A lot of false positives, since the input is 160x90 we lose a lot of detail. The detector loves to pick up random noise as faces.
2. The stride is too big, we lose a lot of detail.
3. Cameras are friends only, you can only show this to your friends.
4. SVMs needs to store data of the hyperplane that does the classification. So more training = more complex hyper plane = lag in game.

## Unity and VRChat Setup

1. Download the latest Face and Object Detection.unity in Release
2. Import into Unity
3. Look in the Prefabs folder
4. Put Face_Object Detect.prefab on your avatar
5. If you want to see what it's doing, put Preview.prefab on your avatar
6. Disable them by default and add a gesture to enable it.

The default detector is trained to look for faces.

## Training Setup

Ignore this if you just want to do use the default detector.

#### Windows 10 64 bit machines
1. Download the latest VRC-SVM Train.exe in Release
#### Anything else
1. Compile the .cpp source code
  - **Requirements**
    - [dirent.h](https://github.com/tronkko/dirent)
    - [OpenCV 4.0.1](https://opencv.org/releases/)
    
<img src="https://i.imgur.com/KDt9mzd.png" align="right" />

2. Make sure the folders are setup the same as the following
3. Run VRC-SVM Train.exe and tell it the folder containing the Positive and Negative training folders using  ```-d```
  - Example:```"VRC-SVM Train.exe" -d="D:\GitHub\Face-and-Object-Detection-in-Unity-Cg\C++\Training Data\Faces"```
  - ```-auto``` will do k-fold cross validation on the training set. **Warning: It crashes a lot if you use** ```-auto```
4. (Optional) To test a detector use ```-t``` and ```-fn``` to tell the program which detector you want to use.
  - Example:```"VRC-SVM Train.exe" -d="D:\GitHub\Face-and-Object-Detection-in-Unity-Cg\C++\Training Data\Faces" -t -fn="D:\GitHub\Face-and-Object-Detection-in-Unity-Cg\C++\out.yaml"```
5. Once training is done, drag the .yaml file into Unity. If you didn't pick a name, the default name is out.yaml. This file is created in the same directory as the .exe

<img src="https://i.imgur.com/PPfXPXU.png" align="right" />

6. Bake the data inside the .yaml file into an image by navigating to Tools -> SCRN -> Bake Support Vectors in Unity

<img src="https://i.imgur.com/L8K5zwg.png" align="left" />

7. Inside the prefab you placed, locate the materials called **Kernel** and **Classify**

Contact me on discord if you have any questions or suggestions: **SCRN#8008**
