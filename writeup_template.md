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

#### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to 
####use some other method and submit a pdf if you prefer.

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit 
####your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes
####the project code.

You're reading it!
here is a link to my [project code](https://github.com/shalaka7/udacity_project_2/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or 
####pandas methods rather than hardcoding results manually.

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

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques?
####Consider including images showing the output of each preprocessing technique.

Pre-processingis very important aspect in this project. Every image is converted into gray scale image then ,
The image data should be normalized so that the data has mean zero and equal variance. For image data, (pixel - 128)/ 128 
is a quick way to approximately normalize the data and can be used in this project. Pre-processing refers to techniques 
such as converting to grayscale, normalization, etc. It also helps to reduce training time.Normalizing the data to the range (-1,1)
In preprocessing, we do many of function like translate, scaling, wraping, brightness these all function help us to 
identify or classify signals very carefully.



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.)
####Consider including a diagram and/or table describing the final model.

My final model consisted of 5 layers. Input is 32*32*3 RGB image. convolution is 1*1 stride [1,1,1,1] with same padding 
we get output 32 *32*32.
we followed by 5 layers.maximum pooling with 2*2 stride we get output 16*16*64.All layer are fully connected

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of
####epochs and any hyperparameters such as learning rate.

To train the model, 
I used an batch size with 100,
epochs is 60,learning rate is 0.0009,
mu is 0 and sigma: 0.1


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. 
####Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. 
####Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and
####why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case,
####discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94%
* test set accuracy of 75%

####If an iterative approach was chosen:
####(1)What was the first architecture that was tried and why was it chosen?
I have used LeNet architectureas per suggestion of lectures

####(2)What were some problems with the initial architecture?
While working on LeNet architecture ,accuracy of model not giving proper decision enough due to image 
data was not pre-processed also the model overfitting data which result in poor validationn accuracy. after preprocessing 
image it gives us good validation accuracy.

####(3)How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model 
####architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing 
####the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. 
####A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both 
####sets indicates under fitting.
 Introduced dropout at first convolution layer, at first Fully connected layer and at last Fully connected layer, dropout
layer is play very important role to avoid overfitting of data and for proper decision required accuracy for validation 
data set. It was high 0.7 to avoid underfitting. 

####(4)Which parameters were tuned? How were they adjusted and why?
Learning Rate, Batch Size, Epoch, Keep probability for dropout.
Learning Rate:- Higher Learning rate train model faster but stagnant earlier than acheving its full potential, 
whereas for lower learning rate model train slower but it achieves lowest possible loss for that model. in our model 
learning rate of 0.0009 yields better results.
Batch Size:- We split model in batches, calculate all parameter for each batches and cascade it for complete model,  
we can deside batch size, in our model Batch size is 100.
Keep Probability:- To avoid overfitting in model we have to introduced dropout at different layer, 
In our model I have added dropout at Conv1 layer, but with different keep probability, like at earlier layer it is 
higher 1.0 in our model


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what
#### quality or qualities might be difficult to classify.

Here are eight German traffic signs that I found on the web:

![alt text][image1]
 As image1 is of  Go straight or left it may confuse due to arrows as same as Roundabout mandatory. 

![alt text][image2]
 As image2 is of no entry the outerpart of sign is main challenge.

![alt text][image3]
 As image3 is of speed limit (20km/h) here the speed limit no is task for correct choice.

![alt text][image4]
  Turn left ahead in image4 detection of turn is important.

![alt text][image5]
 No U turn	in the image5 but sign but detection of no turn is difficult.

![alt text][image6]
 As image3 is of speed limit (60km/h) here the speed limit no is task for correct choice.

![alt text][image7]
 As image3 is of speed limit (80km/h) here the speed limit no is task for correct choice.

![alt text][image8]
 As image3 is of speed limit (40km/h) here the speed limit no is task for correct choice.
 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
####At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to 
####the accuracy on the test set

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

Test Set Accuracy = 94 % (original test set)
Test Set Accuracy = 75 %  (main test set)
So compared acuuracy is about 80 %


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax 
####probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type 
####of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also
### be provided such as bar charts)


For the first image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| Go straight or left							|
| .0     				| Roundabout mandatory  						|
| .0					| Speed limit (30km/h)							|

Here the Roundabout mandatory is totally uncertain prediction but in top five predictions.. gave us correct prediction.

For the second image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| No entry							            |
| .0     				| Stop   									    |
| .0					|Keep right							            |

To find No entry sign is very certain which found in first  predictions. only.

For the third image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .35         			| Speed limit (30km/h)							|
| .34     				| Speed limit (20km/h)   						|
| .32					| Speed limit (80km/h)							|

Here to find speed limit is 20km/h it found in second softmax probability but it may not sure because of font difference. 

For the fourth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100         			| Turn left ahead							    |
| .0     				| Keep right  									|
| .0					| Go straight or right							|

Turn to left is very certain so it predictied in first attempt .

For the fifth image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .100      			| Road narrows on the right						|
| .0     				| Speed limit (30km/h)							|
| .0					| Go straight or right							|

No U turn sign is little similer to road narrows to right and it is in top five predictions.

For the sixth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .67         			| Dangerous curve to the right					|
| .32     				| Speed limit (80km/h)							|
| .1					| End of no passing by vehicles over 3.5 metric ton|

speed limit is 60km/h is easy to find but to find limit is 60 which is unpredictible and uncertain but it mostly available in 
top five predictions.

For the seventh image,

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .72         			| Road work							            |
| .27    				| Speed limit (30km/h)				            |
| .0					| Turn right ahead		 			            |


speed limit is 80km/h but due to img quality it may differ and confuse between all speed limit nos.

For the eighth image ...

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .96         			| Speed limit (30km/h)							|
| .2     				| Stop 									        |
| .1					|Road narrows on the right					    |

speed limit is 40km/h here but it is not that clear so it is confused but it present in top five predictions.

In code ,cell no .141 show it properly and however the due to some reason unpredictibility occurs but the predicted output 
is present in top five predictions.