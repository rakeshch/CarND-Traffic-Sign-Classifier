## Project: Traffic Sign Recognition Classifier

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

My complete project code with results for individual steps can be found [here](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. After loading the pickled data for traning, validation and testing, I have used numpy to print out the basic summary of the data set I will be working on in this project.	

* Number of training examples = 34799
* Number of validation examples = 4410
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

#### 2. Exploratory visualization of the dataset.

Bar chart below shows the disribution of classes in the trainign data set.

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/histogram_label_frequency.png)

### Design and Test a Model Architecture

I used the LeNet-5 implementation with preprocessing techniques to obtain a better accuracy on training and validation dataset.

I spent lot of my time working on preprocessing images based on this [paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) by Yann Lecun, but I was not successful in getting it done (commented some of the unsuccessful code used for this approach). I then have to move on with other techniques due to the time constraint.

Below are the following preprocessing steps I applied to every image in the dataset:

* Convert image to grayscale
* Scale, crop and rotate the image
* Apply global contrast normalization

Based on the images that I looked at from the dataset, I felt that there wasn't going to be a whole lot of additional information gained from rgb versus grayscale, so grayscaling could make the neural network faster and potentially help it focus more on what was important.
Then I applied scaling by a factor of 1.8 which resulted in image resolution increase from 32x32 px to 58x58 px. Since traffic signs on raw images do not cover images from edge to edge I applied cropping to bring back the resolution to 32x32 px. This also avoids processing pixels that are not useful, which is beneficial for CNN training speed. As a last step, I applied Global contrast normalization to center each image around its mean value.

 All operations are called in the preprocess-> img_transformation functions.

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/after_preprocessing.PNG)

After preprocessing the dataset, I used LeNet5 as my final model. One thing I did change from the basic LeNet5 structure was adding dropout to the fully connected layers. The layer uses 70% dropout and experiments shown that it really help to decrease overfitting.

My final model consisted of the following layers:

| Layer         		|     Description	        						| 
|:---------------------:|:-------------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   						| 
| Convolution 5x5     	| 1x1 stride, 'VALID' padding, outputs 28x28x6 		|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 						|
| Convolution 5x5	    | 1x1 stride, 'VALID' padding, outputs 10x10x16 	|
| RELU					|													|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 						|
| Flatten	        	| outputs 400 										|
| Fully connected		| outputs 120  										|
| RELU					|													|
| Dropout				| keep probability = 0.7 							|
| Fully connected		| outputs 84  										|
| RELU					|													|
| Dropout				| keep probability = 0.7 							|
| Fully connected		| outputs 43 logits  								|
|						|													|

The CNN was trained with the Adam optimizer.  I tried a few different batch sizes, but settled at 100 as that seemed to perform better than batch sizes larger or smaller than that. I ran only 30 epochs, primarily as a result of time and further performance gains, as it was already arriving at nearly 97% validation accuracy, and further epochs resulted in only marginal gains while continuing to increase time incurred in training. 

As the CNN with 2 convolutional layers, 3 fully connected, a learning rate of .0009 and a batch size 30 appears to result in the optimal CNN, I utilized this for the final model.

My final model results were:

* Validation set accuracy of 96%-97%
* test set accuracy of 95%

### Test a Model on New Images

I choose five German traffic signs found on the web to test my model.
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/web_test_signs.PNG)

Below are the images after preprocessing
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/web_preprocessed_signs.PNG)

Below are the images after running the predictions on preprocessed images using my model.
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/test_images_model_out.PNG)

We can see that Road Work signal is misclassified as General caution, this could be due to the preprocessing step crops the image a bit too much making it difficult for the classifier.

The model was able to correctly predict 4 out of 5 traffic signals, with an accuracy of 80%.

### Top 5 Softmax Probabilities For Each Image Found on the Web

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/Out_images/softmax.PNG)

For the second, third and fourth images,the model is more confident on the speed limit, yield and stop signs than anything it thought on the others.

For the first and fifth sign, the model incorrectly classifies left turn sign as right turn and Road work sign as General Caution. I believe this is due to the way the image has been preprocessed.

So, my model only worked 60% of the additional pictures. I believe with some tweaking of preprocessing of the images I could get better predictions. 
