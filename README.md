
## Project: Build a Traffic Sign Recognition Program

**Build a Traffic Sign Recognition Project**

### The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Extending data set to make better model
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./sample_images/keepleft.png "Merge Sign 1"
[image5]: ./sample_images/circulate.png "Traffic Sign 2"
[image6]: ./sample_images/km_30.png "Traffic Sign 3"
[image7]: ./sample_images/stop1.png "Traffic Sign 4"
[image8]: ./sample_images/row.png   "Traffic Sign 5"
[image9]: ./cross_entropy.png      "Cross entropy Result"
[image10]: ./model_graph.png      "Model Architecture"
### Data Set Summary & Preprocessing

#### 1.Data Set Summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2.Preprocessing
* On the preprocessing step I tried two different method. 
    (1) normalize data after gray scaling.
    (2) first gray scailing and then normalize data
* In my experience first method show much better performance

* Also I compare two python function to normalize image
    (1)sklearn.preprocessing.normalize
    (2) cv2.equalizeHist 
* Two function give similar result but cv2.equalizeHist was much more comfirtable to implement on the image data
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Model searching

* I tried to compare distinct model structure based on 10 epoches results. many stuck at local optimum in first 10 steps so it actually help identifying good models. 

* I considered overall structure of model, Location of dropout layer, implementing batch norm layer, implementation of inception layer 

* I got idea of final model in Pierre Sermanet and Yann LeCun's research

#### 2. Slim
* I found out we can easily implement batch normalization and other technique with slim library of tensorflow


#### 3. Final Model
![alt text][image10]

The code for my final model is located in the 16th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					   | 
|:---------------------:|:-----------------------------------------------: | 
| Input         		| 32x32x1 processed image   					   | 
| first Convolution    	| Valid padding,activation and pooling             |
| Second Convolution    | Valid padding,activation and pooling,connected   |
| conv layer beta       | max pooling first convolution                    |
| conv layer gamma	    | connect cnn filter to conv layer beta			   | 				
| fc_0           	    | layer connecting cnv_layer gamma and second cnn  |
| fc_1           	    | fully connect fc_0      						   |
| fc_2          		| fullly connect fc_2        					   |
| fc_3   				| Final layer   							       |

I used tf.concat to connect the two seperate convolutional layers


#### 4. Model Accuracy 
The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of 90.3%
* validation set accuracy of 93.1 %
* test set accuracy of 92.1%


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 
![alt text][image5]
![alt text][image6] 
![alt text][image7] 
![alt text][image8]

The first image might be difficult to classify because The 30km/h limit image and Round -about Mandatory image contain getty images label and some number on it which might give wrong answer if my model cannot catch key characteristic of image. Overall images on the web is much more brighter than our original images. 

#### 2.

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| Round-about Mandatory | 30khm limit 									|
| Yield					| Yield											|
| 60 km/h	      		| Non-entry   					 				|
| Right of way at next. | Right of way at next.   						|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 
It is much worse than the old test-set result. 
#### 3. Predicted Probabilty on test data set
![alt text][image9] 

* We can verify that our model classify correctly  keep left image, Right of way at the next intersection image and stop image

* My model give wrong answer on RoundaboutMandatory as 30 km/h limit image but it also consider original correct answer with high probability

* Our model is completly wrong on the speed 60 km/h limit sign 

* We can verify that our model get wrong answer on image which have getty images label. We can verify that the part which human's ignore can make crucial difference to the model.

* The part behind label might take crucial point to classify image. 

### Mistake I made. 


#### 1. Shuffle images during training model . 
* Shuffling images during training within epoch will stop your model to improve. I fight my-self to handle this problem. 


#### 2. Save model in same directory. 
* If you save model using saver object in tensorflow it create three files containing checkpoint. unlike other file checkpoint is created in same name. It will be better for you to save each model in different directory to 

#### 3. Using CPU. 
* Thinking of other practice we did with Le-Net and other model I think I can make a model can be trained perfectly with-in few epoches. How ever It was not. 