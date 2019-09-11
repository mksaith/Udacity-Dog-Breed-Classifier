[//]: # (Image References)

[image1]: Images/sample_dog_output.png "Sample Output"
[image2]: ./Images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./Images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: Images/ImageBoundingBox.png "Sample Bounding box"


## Project Overview

Built a pipeline to process real-world, user-supplied images.  Given an image of a dog, algorithm will identify an estimate of the canine’s breed.  If supplied an image of a human, the code will identify the resembling dog breed. The Jupyter Notebook can be found [here](https://github.com/mksaith/Udacity-Dog-Breed-Classifier/blob/master/dog_app.ipynb).

![Sample Output][image1]

## Detect Humans
- Used OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.
- Get bounding box for each detected face.

![Sample Bounding box][image4]

## Detect Dogs
Built DogDetector using pre-trained VGG-16 model, to find the index corresponding to the ImageNet class for a given image.

## CNN from scratch to Classify Dog Breeds
- Performed augmentation using **OpenCV** and **Torchvision**
- The model takes some inspiration from VGG network.
  - Convolutional and Linear layers are initialzed using kaiming normalization. BatchNorm2d weights are initialized to 1 and bias is initialize to 0.
  - The model has 3 sets of convolutional layers.
  - In each layer, there are 2 convolution operations, 2 max pool operations , 1 Batch normaliztion and in last Leaky ReLU activation.
  - I pass the data to fully-connected layer using `AdaptiveAvgPool2d`.
  
- **Learning rate:** Used [The Cyclical Learning Rate technique](http://teleported.in/posts/cyclic-learning-rate/), which improved the performance while training the model instead of fixed learning rate.

- Used **Adam** optimizer which performed better then **SGD**
- Accuracy achieved up to 50% with 100 epochs

## CNN using Transfer Learning to Classify Dog Breeds
- Performed augmentation using **OpenCV** and **Torchvision**

- Used RESNET-50 for transfer learning.  

- **Learning rate:** [The Cyclical Learning Rate technique](http://teleported.in/posts/cyclic-learning-rate/)

- Used **SGD** optimizer

- Accuracy achieved up to 81% with 10 epochs

