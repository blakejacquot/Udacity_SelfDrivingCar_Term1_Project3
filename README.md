# Overview
This is the submission for the Udacity self-driving car class, Term 1, Project 3 (Transfer
Learning)

## Background information
Example use:
python drive.py drive.py model.json

CSV file structure:
Center image, left image, right image, steering angle, throttle, break, speed

Collection of training data is organized into IMG folder (containing all images) and
driving_log.csv.

## Modules
### preprocess.py
Preprocesses image and csv data for use in training.

### model.py
Use preprocessed image and csv data to train a model

### drive.py
Set up to run with the simulator. Includes import of 'preprocess.py' so that images from
simulator can be processed with same pipeline as training data. Note, the default setting
for 'throttle' is 0.2. My trained model mostly works at this throttle but goes off the track
at about 3/4 to the end. Since Udacity has no stated requirement for this value, I've reduced
it to 0.1. At this throttle, my model completes the track and is pretty smooth while doing it.

## Preprocessing
The file 'preprocess.py' executes a pipelined approach after reading in CSV file and
finding the images. First, images are cropped to remove all non-road data. This mostly
involves cropping out the upper portion of the image. Second, the images are converted from
RGB to grayscale. This helps with speed of training and prevents overfitting to the colors
of a scene. Third, the images are normalized from 0/255 to -1.0/+1.0.

Example of cropped image:
![Simulator Screenshot](/fig1.png)

Example of grayscale image:
![Simulator Screenshot](/fig2.png)


## Model Architecture and Training Strategy

### Model architecture
The model more or less follows the [NVidia][Nvidia] architecture. It has 5 convnets followed by
flattening followed by 3 fully-connected layers. The final output is a single value
representing the steering angle. The number of outputs, kernel sizes, and strides were
chosen to match the Nvidia architecture. The only exception is the final convnet where
kernel size was chosen to be 3x1 vs 3x3. This is because the output of the prior layer
did not allow for 3x3, presumably because I'm starting with a different image size.

The Nvidia architecture can be seen here:
![Simulator Screenshot](/fig3.png)

Perhaps different than Nvidia architecture, I chose a 20% dropout rate and Relu activation
after each convnet. Dropout helps prevent overfitting. Activation breaks linearity in
the network and allows learning more complex functions than linear regression.

### Train/val/test splits
Just before ending, 'preprocess.py' executes the train/test/val split on a dataset. The
split is first a 80%/20% train/test split. Of the remaining training data, there is another
80%/20% train/val split.

### Model parameters
Optimizer: Use the Adam optimizer.

Learning rate: Start with default 0.001 rate from Adam optimizer and change to 0.0001 as
we fine-tune the model.

Batch size: Larger batch size is more efficient from parallel computing perspective, but
it reduces sample variance and increases risk of overfitting.

Epochs: Monitor loss calculation to ensure it decreases over epoch. This indicates, we
haven't reached the minima. In my approach of using multiple datasets, the loss always
decreased because there's so much variance in the data. I used the number of epochs to
over/underweight a given dataset.

### Datasets
'BCJ Udacity': 5100 training samples, unknown number of laps or driving style
'BCJ_dataset1': 1900 training samples, 2 laps of normal driving
'BCJ_dataset2': 2000 training samples, 2 laps of slightly drunk driving
'BCJ_dataset3': 4200 training samples, 3 laps of drunk driving
'BCJ_dataset4': 500 training samples, special situations

In the above datasets, 'drunk driving' represents careening towards edges of track and
recovering. This helps the car recover from extreme situations. The special situation
dataset represents extra samples where the track differs in character and provides extra
training here.

### Training notes
I found good results by training with 10-20 epochs for each dataset. I trained with 'Udacity'
first, and then each dataset in order. I reduced the epochs and decreased the learning rate
with each training session.

### Discussion

How did I decide on batch size?


How did I decide on number of epochs?


Did I test different types of networks to see how results are affected?

#### Details of architecture:
The characteristics of the architecture.
The type of model used.
The number of layers in the model.
The size of each layer.
Kernel sizes and strides for your layers used.
Visualize architecture with model.summary() in Keras.



Layer (type)                     Output Shape          Param #     Connected to
convolution2d_1 (Convolution2D)  (None, 38, 158, 24)   624         convolution2d_input_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 38, 79, 12)    0           convolution2d_1[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 38, 79, 12)    0           maxpooling2d_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 38, 79, 12)    0           activation_1[0][0]
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 17, 38, 36)    10836       dropout_1[0][0]
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 17, 19, 18)    0           convolution2d_2[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 17, 19, 18)    0           maxpooling2d_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 17, 19, 18)    0           activation_2[0][0]
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 7, 8, 48)      21648       dropout_2[0][0]
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 7, 4, 24)      0           convolution2d_3[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 7, 4, 24)      0           maxpooling2d_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 7, 4, 24)      0           activation_3[0][0]
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 5, 2, 64)      13888       dropout_3[0][0]
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 5, 1, 32)      0           convolution2d_4[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 5, 1, 32)      0           maxpooling2d_4[0][0]
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 5, 1, 32)      0           activation_4[0][0]
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 3, 1, 64)      6208        dropout_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 3, 1, 64)      0           convolution2d_5[0][0]
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 3, 1, 64)      0           activation_5[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 192)           0           dropout_5[0][0]
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           19300       flatten_1[0][0]
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]
Total params: 78075

