## Project: Traffic Sign Recognition Classifier

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set 
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

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

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/histogram_label_frequency.png)

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

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/after_preprocessing.PNG)

After preprocessing the dataset, I used LeNet5 as my final model. One thing I did change from the basic LeNet structure was adding dropout to the fully connected layers. The layer uses 70% dropout and experiments shown that it really help to decrease overfitting.

The CNN was trained with the Adam optimizer, batch size = 100 images, learning rate = 0.0009. The model was trained for 50 epochs (34799 images in each epoch) with one dataset. Learning rate was updated by try and error process.

Validation Accuracy: 97%

Test Accuracy: 95%

### Test a Model on New Images

I choose five German traffic signs found on the web to test my model.
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/web_test_signs.PNG)

Below are the images after preprocessing
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/web_preprocessed_signs.PNG)

Below are the images after running the predictions on preprocessed images using my model.
![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/test_images_model_out.PNG)

We can see that Road Work signal is misclassified as Speed imit (70km/h), this could be due to the preprocessing step crops the image a bit toomuch making it difficult for the classifier.

The model was able to correctly predict 4 out of 5 traffic signals, with an accuracy of 80%.

### Top 5 Softmax Probabilities For Each Image Found on the Web

![Screenshot](https://github.com/rakeshch/Traffic_Sign_Classifier/blob/master/softmax.PNG)

For the first image, the model correctly preicts the Turn left ahead signal. But, given the highest probability was only around 7% for left turn and 5% for right turn, the model definitely struggles distinguishing between left and right signs.

For the second, third and fourth images,the model is more confident on the speed, yield and stop signs than anything it thought on the others.

For fifth, it incorrectly guesses Road work sign. I believe this is due to the way the image shaped after preprocessing. 

So, my model only worked 80% of the additional pictures. I believe with some tweaking of preprocessing of the images I could get better predictions. 
