#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./writeUpImages/graph.png  "Graph"
[image2]: ./writeUpImages/before.png "Before"
[image3]: ./writeUpImages/after.png  "After"
[image4]: ./writeUpImages/images(1).jpg "Traffic Sign 1"
[image5]: ./writeUpImages/images(3).jpg "Traffic Sign 2"
[image6]: ./writeUpImages/images(4).jpg "Traffic Sign 3"
[image7]: ./writeUpImages/images(6).jpg "Traffic Sign 4"
[image8]: ./writeUpImages/images(7).jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

###Data Set Summary & Exploration

####1.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the python and numpy libraries to calculate summary statistics of the traffic
signs data set:

* The size of training set is 347999
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data how many images are 

in each of the 43 classes

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the 4th and 5th code cells of the IPython notebook.

As a first step, I decided to convert the images to grayscale while keeping the originals, then split originals in b, g, and r single channel images. I applied Histogram Equalization to all 4 images, then later merged them in a single 4 channel image BGRG.

Here is an example of a traffic sign image before and after processing.

![alt text][image2]

![alt text][image3]

As a last step, I normalized the image data because it improved my accuracy and also reshaped the image data, to (32,32,4).


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x4 BGRG image   							| 
| Convolution 3x3     	| 1x1 stride, Valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Outputs 14x14x6   				|
| Dropout   	      	| keep prob .88                  				|
| Convolution 3x3       | 1x1 stride, Valid padding, outputs 10x10x16  	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Outputs 5x5x6      				|
| Dropout   	      	| keep prob .88                  				|
| Flatten               | Outputs 400                               	|
| Fully Connected		| Outputs 120									|
| RELU					|												|
| Dropout				| keep prob 0.88								|
| Fully Connected		| Outputs 84									|
| RELU					|												|
| Dropout				| keep prob 0.88								|
| Fully Connected		| Output 43										|
| Logits				|												|

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 7th cell of the ipython notebook. 

To train the model, I used AdamOptimizer with a batch size of 128, 85 Epochs, and learning rate at 0.0015255

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 8th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.8% 
* test set accuracy of 94.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
I choose to use a modified version of LeNet using Max Pooling after both convolutional layers and Dropout of 0.88 after every layer except the last.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

They might be difficult to classify because the process of resizing the images to (32,32) results in image loss.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 10th and 11th cells of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Bumpy Road      		| Bumpy Road   									| 
| Ahead Only     		| Ahead Only									|
| Stop Sign				| Stop Sign										|
| 70 km/h	      		| 70 km/h   					 				|
| 50 km/h   			| Left Turn         							|


The model was able to correctly guess 4 of the 5 traffic signs here, but with the rest of the images it gives an accuracy of 73.3%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 12th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Bumpy Road   									|
| .00014    			| General Caution								|
| .00010				| Traffic Signals								|
| .0000028	      		| Wild Animal Crossing 			 				|
| .000000071		    | Slippery Road     							|

