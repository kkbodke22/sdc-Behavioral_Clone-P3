Project Description:

*Behavioral Cloning Project**

In this project, I use a neural network to clone car driving behavior. 

The network is based on The NVIDIA model, which has been proven to work in this problem domain.

The goals of this project are the following:

* Use the simulator to collect data of good driving behavior.
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

My project includes the following files:

model.py:
    The script used to create and train the model.
    
drive.py:
    The script to drive the car provided by the udacity.
    
model.h5:
    Containing a trained convolution neural network.
    
README.md


Model Architecture:

First the data is normalized in the model using a Keras lambda layer.

Training data was chosen to keep the vehicle driving on the road. 
I used a combination of center lane driving, recovering from the left and right sides of the road.

My model consists of a Convolution Neural Network with 3 5x5 filter sizes followed by 2 3x3 filter sizes.

The model then includes RELU layers to introduce nonlinearity.

Then adam optimizer, so the learning rate was not tuned manually 

The model was trained and validated on different data sets to ensure that the model was not overfitting 
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 
