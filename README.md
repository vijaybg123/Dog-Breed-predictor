# Dog-Breed-predictor

### Table of contents
1. [Project Overview](#Overview)
2. [Requirements](#req)
3. [Installation](#install)
4. [Files Description](#file)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Blog](#blog)
8. [Credits and Acknowledgements](#ack)

## Project Overview <a name="Overview"></a>
The goal of a project is to train an effective dog breed predictor using CNN (Convolutional Neural Networks) using 8,351 dog images of 133 breeds. In this project, we build a dog and a human face detector. 
In the first attempt, we build a CNN model from scratch, where we expect an accuracy of at least 2%.
In the second attempt, we use pre-trained model VGG16 to create a predictor, and to the knowledge, we are expecting accuracy of above 20%, if we are able to get anything above 60% accuracy, we will use the same model to predict dog breed.
If we fail to achieve the accuracy of anything above 60%, then in the third attempt, we use the transfer learning technique with any one of the bottleneck features among VGG19, ResNet 50, Inception or Xception. I will be using ResNet 50 bottleneck feature.

## Requirements <a name="req"></a>
Python3 or Jupyter Notebook 
Optional : GPU (better than CPU when training images)
##### libraries: 
* opencv-python
* h5py
* matplotlib
* numpy
* scipy
* tqdm
* scikit-learn
* keras
* tensorflow
* jupyterlabs
* Sklearn
* pandas
* CV2
* PIL

## Installation <a name="install"></a>
1. Install Python 3 or Jupyter Notebook and install the following libraries.
2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-prooject/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

## File Description <a name="file"></a>
The files and folders used in this projects:
1. haarcascades/haarcascade_frontalface_alt.xml: Viola-Jones face detector provided by OpenCV.
2. saved_models/weights.best.InceptionV3.hdf5: Inception v3 model trained using transfer learning.
3. dog_app.ipynb: a notebook that explains the classifier code.
4. extract_bottleneck_features.py: functions to compute bottleneck features of image.
5. images/: contains few random images

## Results <a name="results"></a>
1. saved_models/weights.best.from_scratch.hdf5: 4.78%
2. saved_models/weights.best.VGG16.hdf5: 38.76%
3. saved_models/weights.best.Resnet50.hdf5: 76.19%

## Conclusion <a name="conclusion"></a>
Our goal was to train a model that is effective in predicting the dog breed. So we tried training a model by building CNN layers from scratch with three convolutional layers with a kernel size of 2 and activation layer 'relu.'. We also used the max-pooling pooling layers between them with the pool size of 2. But yet we were able to achieve an accuracy of 4.78%, which is negligible in predicting anything.

To know more about the project 

Later we used the pre-trained model 'VGG16.' Which is already trained on millions of images, but still, we were able to get an accuracy of 38.76%, which is not satisfactory to predict the dog breed.

At last, Thanks to the transfer learning method and pre-trained model ResNet50 bottleneck feature, we were able to get the accuracy of 76.19%, which is indeed satisfactory.

Finally, I can say that there are few other ways to improve the model. The possible ways can include 
1. The dataset is small to train an effective model, so with more image data, we can improve the training model.
2. Hyperparameter tuning can enhance performance. The parameters can include learning rates, dropout and batch size.
3. An ensemble of models.

At last, I would like to thank Udacity for an opportunity to work on this project and enhance our skills.

## Blog<a name="blog"></a>
All the necessary analysis and explaination is found on [medium](https://medium.com/@coolvijaygowdachintu/dog-breed-identification-app-using-convolutional-neural-network-cnn-778dafb05f03)

## Credits and Acknowledgments <a name="ack"></a>
All the data and associated documentation files belongs [udacity](https://www.udacity.com/)
