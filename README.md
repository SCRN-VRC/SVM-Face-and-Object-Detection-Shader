<img src="/Images/Project.png" align="middle" width="3000"/>

# Face and Object Detection Shader

A binary classifier implemented with a support vector machine (SVM) written in C++ and converted to HLSL to be used inside VRChat. I have included a program that allows you to train your own detector. You can use it to detect other objects as well.

## Overview
<img src="/Images/SVM.png" align="middle" width="3000"/>

My implementation has three camera inputs to detect varying sizes 480 x 270, 240 x 135, 120 x 67. A 64 x 64 sliding window is applied with a stride of 16 pixels for the largest camera inputs to 8 pixels for the smallest.

HOG features are extracted by binning by the magnitude of the gradient according to the direction of the gradient into 8 bins. The features are normalized to account for differences in lighting.

The radial basis function (RBF) kernel is applied to each of the 1568 features per 64 x 64 image block and the support vectors. At the end, classification is done by doing the summation of all RBF calculations per 64 x 64 image block.

## Problems

I have fixed most of the problems I listed in the previous implementation. However,
1. In VRChat, cameras are friends only, you can only show this to your friends.
2. SVMs needs to store data of the hyperplane that does the classification. So more training = more complex hyper plane = lag in game.

## Setup

No prerequisites, either clone the repository or download the latest [Release](https://github.com/SCRN-VRC/SVM-Face-and-Object-Detection-Shader/releases) and open it in Unity

## Training

These steps are only for the people who want to train their own detector.
1. Get the svm-detector.exe in [Release](https://github.com/SCRN-VRC/SVM-Face-and-Object-Detection-Shader/releases) or compile it yourself. **The .exe only works with x64 bit machines, anything else you'll have to compile it yourself. Compiling the C++ code requires OpenCV 4.0.1.**
2. Layout your training data like this example for the Faces folder. **All pictures must be 64 x 64 in size.**
<img src="/Images/Folders.png">

3. Run svm-detector.exe with the path to the folder. For example, ```svm-detector.exe D:\Storage\Datasets\Faces 100``` will look for images in the Faces folder and train for 100 iterations.
4. After the program finishes, it will output a ```svm-out.yaml``` file containing the support vectors for classification.
5. To use it inside VRChat, drop the ```svm-out.yaml``` file into the Unity project with this repository.
6. Navigate the menu Tool -> SCRN -> Bake Support Vectors, put in the .yaml file and hit Bake!
<img src="/Images/Bake.png">

7. After a few seconds there should be a new ```SupportVectors.asset``` file in the SVM Detector folder. If a ```SupportVectors.asset``` already exists, it will be overwritten.
8. In the SVM prefab, find the game object called SVM place the new ```SupportVectors.asset``` into the Support Vectors texture slot.
<img src="/Images/Replace.png">

9. Done!

## Contact

Contact me on Discord if you have any questions or suggestions: **SCRN#8008**

Thanks to [Merlin](https://github.com/Merlin-san) and 1001 for making this possible.
