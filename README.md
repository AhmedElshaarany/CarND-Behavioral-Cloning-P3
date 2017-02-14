# Project 3: Use Deep Learning to Clone Driving Behavior

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

[//]: # (Image References)

[image1]: ./examples/cropped_center_2016_12_01_13_30_48_404.jpg "Cropped Image"
[image2]: ./examples/flipped_center_2016_12_01_13_30_48_404.jpg "Flipped Image"
[image3]: ./examples/originaly_center_2016_12_01_13_30_48_404.jpg "Original Image"
[image4]: ./examples/bpp.png "Histogram of Data Before Preprocessing"
[image5]: ./examples/app.png "Histogram of Data Before Preprocessing"

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

python drive.py model.json

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. In the dataProcessing.py file, I provide the code that is used for data augmentation.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with various filter sizes and depths between 16 and 64 (model.py lines 83-103) 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a method that I defined (code lines 48-53). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. The initial learning rate used was 0.0001.

####4. Appropriate training data

I used the training data provided by Udacity, it had data that was chosen to keep the vehicle driving on the road. A combination of center lane driving, recovering from the left and right sides of the road was used to ensure that the vehicle stays on track.

After seeing the images, I noticed that the sky and car hood were part of the captured images by the camera. Since this data is irrelevant to the steering angle, I cropped them out of the images that are fed to the model. Below is an example image before and after cropping

![alt text][image3]
![alt text][image1]

In addition, to improve the driving behavior in cases where the car went off track, I had to visualize the data set, so I plotted the historgram for the steering angles. The histogtam showed that the test data was biased towards the steering angles in the range of -0.1 to 0 as shown below. 

![alt text][image4]

To make the model's behavior more uniform, I filtered out most of the images that provide small variation in the steering angle by randomly choosing images with steering angles close to 0. Also, udacity's data provided left and right images, so to provide more data, I used those images with by adding a steering compenstation factor of 0.2 to steering angle of the left images, and subtracting it from the right images. This lead to the histogram shown below. The data processing was done in the dataProcessing.py file.

![alt text][image5]

Also, to make the model be able to deal with sharp turns efficiently, I flipped images with a steering angle that is larger than 0.2 or less than -0.2 and negated their corresponding steering angle values and added them to the dataset. This was also done in dataProcessing.py

![alt text][image3]
![alt text][image2]

For details about how I used the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a model similiar to the [NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) pipeline. This model proved to work well in a real environment, so I tried to create a miniturized version of it to be able to run it on my laptop that does not have a GPU.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it would have multiple dropout layers with a drop probability of 0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. This led me to augment the training data as explained in the previous section.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 83-103) consisted of a convolution neural network with the following layers and layer sizes:
1- 2x2 Convolutional layer and depth 16 followed by a ReLU activation layer

2- MaxPool layer with kernel 2x2 and stride 2x2

3- Dropout layer with an aggressive 50% drop probability

4- 3x3 Convolutional layer and depth 32 followed by a ReLU activation layer

5- MaxPool layer with kernel 2x2 and stride 2x2

6- 2x2 Convolutional layer and depth 64 followed by a ReLU activation layer

7- MaxPool layer with kernel 2x2 and stride 2x2

8- Dropout layer with an aggressive 50% drop probability

9- Five fully connected layers along with one final dropout layer

####3. Training Process

30% of the data into a validation set. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 60. I used an adam optimizer so that manually training the learning rate wasn't necessary.

####4. Working on Track 2

With some extra preprocessing and model tuning and modification, I was able to create another model that is able to complete both tracks, but with less performance on track 1 than the previous model. The files used for this model were driveT2.py, modelT2.json, and modelT2.h5
