# Udacity_CarND_TrafficSignClassifier
Term1-Project2: Traffic Sign Classifier

## Goals

The primary goal of this project is to implement a CNN to recognize road traffic signs.
In implementing the CNN, the emphasis is to understand the working principles &
steps involved. Specifically, 

* Exploring & visualizing the dataset
* Design, train and testing a model architecture
* Using the model to make prediction on new images
* Analyze the softmax probabilities of the new images


## Data set summary & exploration

For this project we focus on the German road traffic signs. Specifically, we are
using the dataset from _The German Traffic Sign Recognition Benchmark
([GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news))_.

The datasets constitutes of 43 classes and overall collection of ~52000 images.
These images are further split into training, validation & test sets as depicted below

* training set = 34799
* validation set = 4410
* test set = 12630

Each sample is a 32x32 pixel image and the corresponding ground truth labels
for each image is provided in csv file (; separated) with following structure

`<img_number>;<label_number>`

The labels are encoded into integers in the range of 0-42, and the corresponding
mapping is provided in the signnames.csv

Example:

0. Speed limit (20km/h)
1. Speed limit (30km/h)
2. Speed limit (50km/h)
3. Speed limit (60km/h)
4. Speed limit (70km/h)
5. Speed limit (80km/h)
6. End of speed limit (80km/h)
7. Speed limit (100km/h)
8. Speed limit (120km/h)
9. No passing
10. No passing for vehicles over 3.5 metric tons

...etc

Here's a sample visualization of the training images and the histogram of the
training set depicting the spread of traffic signs in the training set.

## CNN Model

We implement the famous [LeNet](http://yann.lecun.com/exdb/lenet/) CNN model for
our traffic sign classifier. Although LeNet was originally designed for
character recognition, we can use it for traffic sign classification as well
since the core principle of this technique relies on recognizing
patters/features in images rather than just matching a fixed set of key features
as seen in other image matching solutions.

### Design/Architecture


As shown in the image above, the CNN architecture consists of the following
layers:

1. INPUT    - 32x32x1
2. CONV     - input=32x32x1  output=28x28x6
   
   filter_size = 5x5
   num filters = 6
   padding = 0
   stride = 1
   bias_size = 6
  
  * RELU
  * POOL  - input=28x28x6  output=14x14x6
      kernel_size = 2x2
      stride = 2
      algo = max

3. CONV     - input=14x14x6  output10x10x16

   filter_size = 5x5
   num filters = 16
   padding = 0
   stride = 1
   bias_size = 16

  * RELU
  * POOL  - input=10x10x16 output=5x5x16
      kernel_size = 2x2
      stride = 2
      algo = max

4. FC       - input=5x5x16   output=120
  * SOFTMAX
5. FC       - input=120      output=84
  * SOFTMAX
6. FC       - input=84       output=43


| Layer | Description       | Params |
|-------|-------------------|--------|
| Input | input samples     | 32x32x1 |
| CONV 1| convolution       | filter_size=5x5, #filters=6, padding=0, stride=1, #bias=6 |
| RELU  | activation        | |
| POOL  | sub sampling      | kernel=2x2, stride=2, algo=max |
| CONV 2| convolution       | filter_size=5x5, #filters=16, padding=0, stride=1, #bias=16 |
| RELU  | activation        | |
| POOL  | sub sampling      | kernel=2x2, stride=2, algo=max |
| FC 1  | fully connected   | input=5x5x16(400) output=120 |
|SOFTMAX| activation        | |
| FC 2  | fully connected   | input=120 output=84 |
|SOFTMAX| activation        | |
| Output| output logits     | input=84 output=43 |

### Data pre-processing

__Grayscale Conversion__
As seen in the above design, the model expects the input samples to be in
32x32x1 format. We therefore convert our input samples from RGB (3 channels
32x32x3) to grascale (1 channel; 32x32x1).

__Normalization__
Generally, it required that all the input data be on the same scale, which
yields in faster convergence...etc benefits. While there are several methods to
perform normalization, we perform a quick normalization on our input samples by
subtracting & dividing each of our grayscale pixels by 28.

The image below shows sample images after pre-processing. column1 is the raw
image sample, column2 is the grayscale and column3 is normalized.

### Training

The model was trained using the following components

* optimizer = Adam Optimizer


