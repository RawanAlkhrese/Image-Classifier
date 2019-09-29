# Image Classifier 
## Prerequisites
- **Python versions 3.0+**
- **Numpy:** to do some calculations on the dataset.
- **Json:** to open and load jason files. 
- **Matplotlib:** to visulize the data.
- **Seaborn:** to visulize the data.
- **PyTorch:** used for loading the data , transformation and building neural networks . 
## Project Motivation:
Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. 
We'll train an image classifier to recognize different species of flowers. We can imagine using something like this in a phone app that tells us the name of the flower the camera is looking at. In practice we'd train this classifier, then export it for use in application. here I built command line application using this model.
 ## About the Data:
 The model trained in [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) which contains 102 flower categories.
 *Image-Classifier folder:*
 ##### Notebook files: 
 - **Image Classifier Project.ipynb:** contains the code of the prject. 
 - **Image Classifier Project.html:**  the HTML version of the code file.
 ##### command line application: 
 pair of Python scripts that run from the command line:
 - **train.py:** train a new network on a dataset and save the model as a checkpoint. 
 - **predict.py:** uses a trained network to predict the class for an input image.
 --------------
 *This project is a part of Udacity's Data Science Nanodegree*
