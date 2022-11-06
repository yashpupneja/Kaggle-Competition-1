## Kaggle Competition: Modified Version of MNIST Dataset for Digit Recognition

This is a modified version of the MNIST dataset. The dimensionality of the image in pixels is 1568 and hence the images are reshaped into 56X28. TThe training set has 50000 examples and the test set has 10000 examples. The labels are in the range 0-18. The competition requires us to recognize the two digits in a single image and then sum them. For example, if the image contains 3 and 5, the label should be 8. 

The dataset is available at https://www.kaggle.com/competitions/classification-of-mnist-digits/data

I have used several models for this task. These include:
- Dummy Classifier
- Logistic Regression
- Random Forest 
- Convolutional Neural Network

The best model is a CNN with several layers. The model is trained for 50 epochs and the accuracy is 98.5%. All the models are saved in the Kaggle_submission.ipynb file. The logistic regression from scratch is saved in the Logistic_Regression.py file. 
