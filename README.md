# Face Recognition and Object Detection in Unity Cg
SVM using HOG descriptors implemented in fragment shaders for VRChat

OpenCV Setup

VC++ Directories
D:\OpenCV 4.0.1\opencv\build\include;$(IncludePath)

C/C++ -> General Additional Include Directories
D:\OpenCV 4.0.1\opencv\build\include

Linker -> General -> Additional Library Directories
$(OPENCV_DIR)\lib;%(AdditionalLibraryDirectories)

Linker -> Input -> Additional Dependencies
opencv_world401d.lib;%(AdditionalDependencies)

Environment Variables -> User Variables
Make sure OPENCV_DIR points to install path

Environment Variables -> Path
%OPENCV_DIR%\bin
