# **Traffic Sign Recognition** 


**Project outline:**
Within this project a convolutional neural network has been implemeted for classification of traffic sign signals with 43 classes.
The outline of the project is:
* Loading the data and visualization
* Data augmentation
* Converting the data to grayscale and normalization
* Neural network architercture
* Trainig the Neural network and evaluating the accuracy on the validation set
* Evaluation of the accuracy of the model on the test data
* Evaluation of the accuracy of the model on the images downloaded from web


[//]: # (Image References)

[image1]: ./Figures/data_set.jpg
[image2]: ./Figures/histogram.jpg 
[image3]: ./Figures/rotation.jpg 
[image4]: ./Figures/scaled.jpg 
[image5]: ./Figures/translated.jpg 
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"


### 1. Loading the data and visualization
Similar to all machine learning approaches the data is splited into training, validation and test sets. The shape of each set can be obtained by using "shape" command from the numpy library.

* The size of training set is : (34799,32,32,3)
* The size of the validation set is : (4410,32,32,3)
* The size of test set is : (12630,32,32,3)
* The shape of a traffic sign image is : (32,32,3)
* The number of unique classes/labels in the data set is : 43

Some of the images from the training data set is shown below with the labels included as in the title:

![alt text][image1]

Having a good understanding of the data distribution can be very helpful to train an efficient model. Therefore, the histogram of the data distribution is shown below:

![alt_text][image2]

As it is clear some the signs (classes) have small number of examples. Therefore, the trained model will be biased towards the signs (classes) with more example. In order to overcome this problem data augmentation can be helpful. With data augmentation, new data will be generated and added to the original training set.
### 2. Data augmentation:
For data augmentation first the classes with less than 800 classes will be selected to generate new data for them. Three different data agumentation methods have been employed to generate new data:

* Random rotation: This function gets an image and rotate it randomly around the center of the image. The random rotation is selected among the angle list of [-15, -10, -5 , 5, 10 ,15]. The output of this function on some of the images from the training set is shown below:

![alt_text][image3]

* Random scaling: This function gets an image and scales it up randomly. The output of this function for some of the examples from the training data set is presented below. The random scaling factor are [1.2,1.4,1.6,1.8].
![alt_text][image4]
* Random translation: This function gets an image and applies a random translation to it
![alt_text][image5]

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8] ![alt_text][image9]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      					|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


