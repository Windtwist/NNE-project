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






