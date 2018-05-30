# **Traffic Sign Recognition** 
[//]: # (Image References)

[image1]: ./test_images/test2/1.jpg "Go straight or left"
[image2]: ./test_images/test2/11.jpg "No entry"
[image3]: ./test_images/test2/5.jpg "Speed limit (20km/h)"
[image4]: ./test_images/test2/6.png "Turn left ahead"
[image5]: ./test_images/test2/7.jpg "No U turn"
[image6]: ./test_images/test2/8.jpg "Speed limit (60km/h)"
[image7]: ./test_images/test2/9.jpg "Speed limit (80km/h)"
[image8]: ./test_images/test2/99.jpg "Speed limit (40km/h)"


## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing
Min number of images per class = 180
Max number of images per class = 2010



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique.

   Pre-processingis very important aspect in this project. Every image is converted into gray scale image then ,The image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 is a quick way to approximately normalize the data and can be used in this project. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. It also helps to reduce training time.Normalizing the data to the range (-1,1)
   In preprocessing, we do many of function like translate, scaling, wraping, brightness these all function help us to identify or classify signals very carefully.





#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of 5 layers. Input is 32*32*3 RGB image. convolution is 1*1 stride [1,1,1,1] with same padding and we get output 32 *32*32.
we followed by 5 layers.maximum pooling with 2*2 stride we get output 16*16*64.All layer are fully connected

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an batch size with 100,epochs is 60,learning rate is 0.0009,mu is 0 and sigma: 0.1


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93%
* test set accuracy of 45%



### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image1]
![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set
Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Go straight or left	| Go straight or left  							|
| No entry    			| No entry      								|
| Speed limit (20km/h)	| Speed limit (30km/h)							|
| Turn left ahead  		| Turn left ahead 				 				|
| No U turn			    | Road narrows on the right 					|
| Speed limit (60km/h) 	| Dangerous curve to the right					|
| Speed limit (80km/h) 	| Road work	     				 				|
| Speed limit (40km/h)  | Speed limit (30km/h)							|


The model was able to correctly guess 4 of the 8 traffic signs, which gives an accuracy of 50%. This compares favorably to the accuracy on the test set of 75%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


For the first image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| Go straight or left							|
| .0     				| Roundabout mandatory  									|
| .0					| Speed limit (30km/h)							|

For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| No entry							|
| .0     				| Stop   									|
| .0					|Keep right							|

For the third image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .35         			| Speed limit (30km/h)							|
| .34     				| Speed limit (20km/h)   						|
| .32					| Speed limit (80km/h)							|

For the fourth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| Turn left ahead							|
| .0     				| Keep right  									|
| .0					| Go straight or right							|

For the fifth image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100      			| Road narrows on the right						|
| .0     				| Speed limit (30km/h)							|
| .0					| Go straight or right							|

For the sixth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .67         			| Dangerous curve to the right					|
| .32     				| Speed limit (80km/h)							|
| .1					| End of no passing by vehicles over 3.5 metric ton|

For the seventh image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .72         			| Road work							|
| .27    				| Speed limit (30km/h)				|
| .0					| Turn right ahead		 			|

For the eighth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96         			| Speed limit (30km/h)							|
| .2     				| Stop 									|
| .1					|Road narrows on the right							|

