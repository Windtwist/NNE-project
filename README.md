# NNE-project
Repository for Neural Network class project - Prediction of calories based on food pictures using NN

## Part 1. Conceputal design

Goal of the project is to desing a Nueral Network where for its input it will take a picture of a plate with food items on it and the output will be the number of calories each item is and what the combined total is.

Dataset - couple of dataset come to mind, but found a dataset <http://www.ivl.disco.unimib.it/activities/food-recognition/> that is perfect for this task.
It consists of many images of different foods on a plate which we can use to identify different foods on a plate. 
For testing I'd split the dataset into 80/20, but the real test would be on an image I could take of a plate and detect if the model would work on those.

The pipeline would consist of the following:

  1. Image is input into the model
  2. Identify food and mask its area -> couple models come to mind but believe mask R-CNN is best
  3. After mask is obtained estimate calories, could be surface area or some other model, open to discussion
     - edge detection of plate and then estimating size of food based on pixel size comparison  
  4. Output is an image with labeled food and each individual calorie of food + a total calorie estimate number.



