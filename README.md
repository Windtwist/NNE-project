# NNE-project
Repository for Neural Network class project - Prediction of calories based on food pictures using NN

## Part 1. Conceputal design

Goal of the project is to desing a Neural Network where for its input it will take a picture of a plate with food items on it and the output will be the number of calories each item is and what the combined total is.

Dataset - couple of dataset come to mind, but found a dataset <http://www.ivl.disco.unimib.it/activities/food-recognition/> that is perfect for this task.
It consists of many images of different foods on a plate which we can use to identify different foods on a plate. 
For testing I'd split the dataset into 80/20, but the real test would be on an image I could take of a plate and detect if the model would work on those.

The pipeline would consist of the following:

  1. Image is input into the model
  2. Identify food and mask its area -> couple models come to mind but believe mask R-CNN is best
  3. After mask is obtained estimate calories, could be surface area or some other model, open to discussion
     - edge detection of plate and then estimating size of food based on pixel size comparison  
  4. Output is an image with labeled food and each individual calorie of food + a total calorie estimate number.


## Part 2. Dataset

The dataset used for training and validation: [UNIMIB2015 Food Database](
http://www.ivl.disco.unimib.it/activities/food-recognition/) - [paper](http://www.ivl.disco.unimib.it/wp-content/files/JBHI2016_r1.pdf)

Dataset used for testing: [Nutrition5k](https://github.com/google-research-datasets/Nutrition5k) - [paper](https://arxiv.org/abs/2103.03375)

### Differences:

Main difference between the dataset that will be used for training and validation is that the training set has many angles of the food and their calorie counts, they also havie a disticnt plate which I will use to estimate the bounding boxes where food can be detected and also use the plate size to determine the calorie amount. Since the plates are a fixed size in tthe image no matter how close/far the food in the image is I can have a size estimator, and calculate the size based on detecting edges for the plates and
calculating the real size based on pixel size in image, sthat way images which can be inputted into the model when testing comes can be of many different resolutions. The testing dataset has many images, but one crucial aspect to see how our model performs is that 
the testing dataset has different plates and sizes, whewreas the training has same size plates and trays. Another addition in the testing dataset is that there are videos which we can input to test our model, to accomplish the end goal of real-time detection of calories.

## Impoprtant Update:

After careful conbsideration and attempting to run the Mask R-CNN model an getting many errors with dependency and version issues from Collab, it was decided to proceed with Yolo V8 for the detection and Image segmentation.
Also to make life easier and not need to manually annotate the dataset, another dataset was found on Roboflow that has the option of already being in the neccessary format for Yolo V8.

New dataset for training and validation: [https://universe.roboflow.com/college-gg4mu/food-image-segmentation-using-yolov5]

## Part 3. First solution and validation accuracy

### Reason for chosen architecture: Yolo V8

The reason Yolo V8 was chosen, is because it is the best model currently when it comes to instance segmentation. Out of the models tested such as Vgg19, InceptionV3 it proved to be the fastest and most accurate model.
And it is also the simplest model to run and if the project is decided to be put to recieve input form video feed it is the best model to do so. The current itteration of the model is trained on 25 epochs, for the last part 
of the project we will train for more, however for the initial solution this is a good representation of the data. 

#### More on YOLO architecture:

1. <ins>Convolutional Layers:</ins> YOLO utilizes a series of convolutional layers to extract features from the input images. The choice of convolutional layers is justified by their proven efficiency in capturing spatial hierarchies in image data, making them ideal for image recognition tasks.

2. <ins>Leaky ReLU Activation:</ins> YOLO often employs Leaky ReLU as an activation function to introduce non-linearity into the model while allowing for a small, non-zero gradient when the unit is not active, helping to prevent dead neurons during training

3. <ins>Composite Loss:</ins> YOLO's loss function includes many components in its loss calculation these inlcude the bounding box regression loss, object loss, and classification loss. This alows the model to simultaneously optimize for accurate localization with bounding boxes, and correct class prediction.

### Evaluation methods - see the Google Collab document:

Different methods were used to evaluate the model, main ones is mAP@50 this is the mean average value on Intersetion over Unit(IoU) this shows us how close the predicted bounding box is to the ground truth and from our model we achieve 98.1% this indicates that our model is 98.1% close to the ground-truth when the prediction between the bounding box and ground truth threshold is set to 50%. 

From the confusion matrix one can incure that that accuracy is decently high since there is an abundance of darker spots than lighter spots on the graph. 

From the results.png image there are many indicators there as well:
  - Loss graph: we can see the loss graphs which show that the training and validation loses are decreasing which suggests the model is learning and can generalize to new data well. 
  - The precision graph: with precision being the ratio of true positives to the sum of true positives and false positives. The higher the precision, the lower the number of false positives. The precision is fluctuating but generally trends upward, suggesting that the model is correctly identifying objects more often as it trains.
  - The recal graph: the ratio of true positives to the sum of true positives and false negatives. The chart shows that the recall increases sharply initially and then plateaus, which means the model is capturing most of the relevant data points.

### Analysis of results:

 - General Improvement: All the metrics and losses show improvement as the number of epochs increases, which is a good indicator that the model is learning and its performance is improving over time.

 - Overfitting Check: Since the validation loss follows the training loss closely and both are decreasing there is no indication that the model is overfiting.

 - Plateaus: Some metrics, especially recall and mAP50-95, show signs of plateauing, which could indicate that the model is reaching its maximum performance capacity.
   
 - Smoothness in Learning: The smoothness in the loss and metric plots is a good indicator that the model is stable.

### Next steps:

Having choosen the correct model based on our initial results, I would want to train it on a bigger database that I found through roboflow that has around 82 classes. [https://universe.roboflow.com/carmine-tarsia/food-detection-ognx8]. After this and checking if the results are satisfactory since it will take a longer time to train. We can implement the calorie counter algorithm. Looking at past papers most of them indicate regression line to be the best at calculating based on volume and energy, must look into databases that have these infomrations on real food items and then check the bounding boxes or image segmentations. Looking for feedback to see if this is a good approach of if I should scrap the YOLO idea and try U-net next, had many issues with Mask R-CNN where it would not work with Collab because one cannot downgrade in collab back to tensorflow versions under 2. 

## Part 4. Final solution

### Test dataset performance
The test dataset is part of the split of the training data, its is around 350 images when comapred to the 699 images in the validation set and 2447 images in the training set. Diffenre is that the pictures are much more varried, and have more food classes on them an avg of 5+ classes. I believe that this is a big difference compared to training set becuase in an actual real-time video streaming application that takes input form a phone camera which was the intended idea behind this project it is suffisive. I do agree however that it is also very generalizable to this dataset, but that is because of the training sample and labels. There are many specific labels for specific Indian and Chinesse fods so generalizing those to a wider culture of foods will not work.

Different methods were used to evaluate the model, main ones is mAP@50 this is the mean average value on Intersetion over Unit(IoU) this shows us how close the predicted bounding box is to the ground truth and from our model we achieve 98.8% this indicates that our model is 98.8% close to the ground-truth when the prediction between the bounding box and ground truth threshold is set to 50%. So it performs 0.7% better than the validation set. Main imrpovements to make, is to train on a larger more diverse dataset, that way my final solution would be mroe generalizable since right now it is limited to the dataset which is why testing accuracy is very high. 

### Final calorie count

In order to get the final calorie count I had to get a little creative and it is the reason why the solution works on a limited amount of simples, specific one being the choosen test sample which can be seen below. Okay so in order to get the caloric values, there are many new programs employed. First, one must find the area of the plate where the food is placed, that was the first challenge. I used canny edge detection and countours with an elipsoid contour to find the area which the plate occupies. After this I took all my detection coordinates, and mapped out the area each food takes in that elipsoid are region. Then those value were normalized to the region, so taht we cannot exceed 100% of that region. Bad solution because we do not know the actual plate size and just estimating it from the elipsoid area compared to the food area. Then I pull an API call using Nutrionix API and find real caloric content of all the foods detected. Then I multiply that normalized area with that caloric content and get my final resutls + I add some margin because the caloric value was too low, because of the way I normalized the food items to the plate. Below you can see that process.

#### Classifier for food with bounding boxes + coordinates

![alt text](/images/classifier.png)

#### Plate detection

![alt text](/images/plate.png)
![alt text](/images/plate2.png)


#### API call - Nutrionix API

![alt text](/images/api call.png)


#### Final calorie estimation

![alt text](http://url/to/img.png)
