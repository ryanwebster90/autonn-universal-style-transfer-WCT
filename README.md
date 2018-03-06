# Universal style transfer via Feature Transforms in autonn

MATLAB implementation of "Universal Style Transfer via Feature Transforms", NIPS 2017 (official torch implementation [here](https://github.com/Yijunmaverick/UniversalStyleTransfer))

# Dependencies

[autonn](https://github.com/vlfeat/autonn) and [MatConvNet](https://github.com/vlfeat/matconvnet)

The VGG-19 encoder and decoder weights must be downloaded [here](https://drive.google.com/open?id=1Ufvv2SV7PQZjEDMJGGJVxeJC4H7yEVg_), thanks to [@albanie](https://github.com/albanie) for converting them from PyTorch.

Run setup.m to add matconvnet and autonn into your path. Then, running the demo files will perform style transfer and texture synthesis.


# Style transfer #

![styletransfer](https://i.imgur.com/udYm9RR.png)

# Texture Synthesis

 bones           |  flowers
:-------------------------:|:-------------------------:
![](https://i.imgur.com/X5XCfht.jpg)  |  ![](https://i.imgur.com/8DlIgej.jpg)
