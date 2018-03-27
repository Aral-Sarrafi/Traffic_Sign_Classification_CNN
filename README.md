# **Traffic Sign Recognition in Tensorflow** 


**Project outline:**
Within this project a convolutional neural network has been implemeted in tensorflow for classification of traffic sign signals with 43 classes.
The outline of the project is:
* Loading the data and visualization
* Data augmentation
* Converting the data to grayscale and normalization
* Neural network architercture
* Trainig the Neural network
* Evaluation of the accuracy of the model on the images downloaded from web
* Suggestions


[//]: # (Image References)

[image1]: ./Figures/data_set.jpg
[image2]: ./Figures/histogram.jpg 
[image3]: ./Figures/rotation.jpg 
[image4]: ./Figures/scaled.jpg 
[image5]: ./Figures/translated.jpg 
[image6]: ./Figures/augmented_data.jpg
[image7]: ./Figures/new_histogram.jpg
[image8]: ./Figures/grayscale.jpg
[image9]: ./Figures/normalized.jpg
[image10]: ./Figures/modifiedLeNet.jpg
[image11]: ./Figures/normalized_web_images.jpg

[image12]: ./Figures/Sign_0.jpg
[image13]: ./Figures/Sign_1.jpg
[image14]: ./Figures/Sign_2.jpg
[image15]: ./Figures/Sign_3.jpg
[image16]: ./Figures/Sign_4.jpg
[image17]: ./Figures/Sign_5.jpg








## 1. Loading the data and visualization
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
## 2. Data augmentation:
For data augmentation first the classes with less than 800 classes will be selected to generate new data for them. Three different data agumentation methods have been employed to generate new data:

###  Random rotation:
This function gets an image and rotate it randomly around the center of the image. The random rotation is selected among the angle list of [-15, -10, -5 , 5, 10 ,15]. The output of this function on some of the images from the training set is shown below:

![alt_text][image3]

### Random scaling:
This function gets an image and scales it up randomly. The output of this function for some of the examples from the training data set is presented below. The random scaling factor are [1.2,1.4,1.6,1.8].

![alt_text][image4]

### Random translation:
This function gets an image and applies a random translation to it. The tranlation is a set of two integer numbers ( one for x direction and the other in y direction) which is selected between -2 and 2.

![alt_text][image5]

### New data set check:
Before proceeding to training the model. I visulized some the new generated data with the labes, to make sure that the data set is consistent. Adding data with incorrect labels will make the model to fail. The figure below show some of the generated data and the labels are matching the sign. Therefore, the new data set is safe to train the model.

![alt_text][image6]

Revisiting the distribution of the new training data set can be helpful to better understand the drawback of the model. The histogram of the new training data set is presented below:

![alt_text][image7]

The number of examples for the selected signs are increased. Ideadlly the distribution of the data should be close to uniform to avoid any sort of biased training. (This part is not investigated in this project)

## 3. Converting data to grayscale and normalization:
The netural network architecture used in this project are LeNet5 and imporved LeNet5 which are both efficient in processing the grayscale images. Therefore, we will be converting the data set to grayscale to compatible with the LeNet5. The grayscale image are shown below :

![alt_text][image8]

Data normalization is also very helpful while so;ving the optimization problem for deep learning. It has been shown that the optimization algorithms such as stochastic gradient decent or Adam optimizer perfrome better on the normalized data, because the cost function will have better symetric proprties in different directions that can be helpful for the convergence of the optimizer. Figure below shows some of the normalized images from the training data set.

![alt_text][image9]

## 4. Neural network architercture:

With in this project I used the modified LeNet neural network architecture addopted from Sermanet/LeCunn traffic sign classification journal article. Moreover, dropout normalization is used in the fully connected layers for reguralization of the model and avoiding overfitting.

![alt_text][image10]

**Network details:**
The details of the neural network architecture can be found in the tensorflow implementation in the ipython file.
## 5. Trainig the Neural network

The neural network is trained by Adam optimizer. The weights and biases of the model are updated by Adam optimizer in a way to minimize the cross entropy cost function. Hyperparameters of the training are set to be:

* learning rate = 0.0009
* EPOCHs = 20
* Batch size = 256
* Keep_prob = 0.5
* Optimizer: Adam optimizer


The performace of the trained model is evaluated on the training, validation and test set:
* training set accuracy of : 0.999
* validation set accuracy of : 0.949
* test set accuracy of : 0.935

## 6. Test a Model on New Images

I downloaded 6 german traffic signs from web which are included in the "German Traffic Signs" folder. As it is expected the images found on the web are not necessarily the same format and size that out model requires. Therefore, the images should be resized and normalized to be compatible with our trained model. Figure below shows these images after being resized using opencv and normalization.

![alt text][image11]

Then the images are feed to the trained model (our model has never seen any of these image)
Here are the results of the prediction obtained from the model:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Wild animals crossing     			| Wild animals crossing 										|
| Speed limit (50km/h)					| Speed limit (50km/h)											|
| Turn right ahead	      		| Turn right ahead					 				|
| Speed limit (60km/h)			| Speed limit (60km/h)      					|
| Stop			| Stop      					|

The trained model was able to identify the new images downloaded from the web accuratly. However, this does not mean that the model will be accurate for all the other images. As it has been shown tha accuracy of the model for test set and validation set is 93% and 95% respectively.

Other than the accuracy of the model in predicting the label for the sign, the certaintity of the model for the predicted model is important. The figures below show the probabilities associated with the first top 5 guesses of the trained model for each of the images downloaded from the web. Hopefully the model is providing the correct labels with high cetrainity close to one for each of the images. However, one can find examples in which the model has less certainity in the predicted labels.

![alt text][image12]![alt text][image13] 
![alt text][image14]![alt text][image15] 
![alt text][image16]![alt text][image17] 

## 7. Suggestions

The trained model has an acceptble performance. However, there is still lots of room for imporovement. Some of the ideas that are worth investigating are :

**Cosidering a better data augmentation approach:**
The data augmentation can be performed is a smarter way which leads to a uniform distribution of the training data set to avoid a biased training of the model. More data augmentation function can be introduced such as bluring the images adding noise or random affine transfomations.

**Fine tunninf the model hyperparameters:**
Number of EPOCHs for training, batch size, learning rate and keep_prob are the hyperparameters of the trained model. Fine tunning any of these hyperparameters can enhance the performace of the model.

**Using other network architecture:**
With in this projec I used modified LeNet for image classification. The drawback of this method is that the color of the images will not be taken into account in the classification procedure. Better image classification networks such as AlexNet or VGG-16 can take advantage of the color in the images which may lead to a better pefromance. However, training these models can take a longer time.
