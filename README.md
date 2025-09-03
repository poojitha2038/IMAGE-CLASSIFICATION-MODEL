# IMAGE-CLASSIFICATION-MODEL

COMPANY: CODTECH IT SOLUTIONS

NAME: VITHANALA POOJITHA

INTERN ID: CTO4DY196

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

Task 3: CNN for Image Classification – Description

The third task of the internship in CODTECH deals with Image Classification through a Convolutional Neural Network (CNN). Image classification is a core problem in computer vision, in which one wants to classify an input image into one of a set of predetermined categories. CNNs are especially effective at this because they learn spatial hierarchies of features automatically from unprocessed image data.

Objective

The goal of this exercise is to create, train, and test a CNN model that is capable of classifying images into the corresponding categories. Through this, one obtains hands-on experience with deep learning methods for vision issues, which have broad usage in the areas of healthcare, autonomous vehicles, security, and e-commerce.

Steps Involved

1. Dataset Loading
For illustration, the CIFAR-10 dataset is typically used. The dataset consists of 60,000 color images in 10 classes like airplane, car, bird, cat, and dog. Every image is 32x32 pixels in size. The dataset is divided into training and test sets, and normalization is done so that pixel values lie between 0 and 1.

2. Constructing the CNN Model
Convolutional Layers: Scrape spatial features from images by using filters.
MaxPooling Layers: Downsample spatial dimensions, retaining significant features.
Flatten Layer: Translates feature maps to a 1D vector for the dense layers.
Dense Layers: Classify based on the learned features.
  The output layer has a softmax activation to make predictions over multiple classes.

3. Model Compilation and Training
The model is trained with the Adam optimizer and categorical cross-entropy loss, appropriate for multi-class classification. Training is done for several epochs with both training and validation data so that the model can learn significant patterns.

4. Evaluation
Once trained, the model is tested on the test dataset. Accuracy, the fraction of images correctly classified, is the major metric used. Furthermore, learning accuracy and loss curves over epochs are graphed to inspect the training and identify problems such as overfitting.

5. Predictions
The learned model can be utilized to predict on unseen images. By comparing predicted values with true labels, we are able to evaluate the trustworthiness of the classifier.
Task 3 illustrates the capability of Convolutional Neural Networks in image classification tasks. With the entire pipeline of dataset preparation, construction of CNN, training, evaluation, and visualization, task 3 gives practical experience in deep learning for computer vision. Although the model constructed here is quite basic, it sets the stage for more sophisticated architectures like ResNet, VGG, or EfficientNet.

OUTPUT


<img width="763" height="599" alt="Image" src="https://github.com/user-attachments/assets/12591d70-ca27-438c-a05d-ddda6dc558e6" />
<img width="749" height="599" alt="Image" src="https://github.com/user-attachments/assets/6d8d3c66-63dd-4bf2-8974-30c218b0f4b7" />
<img width="559" height="604" alt="Image" src="https://github.com/user-attachments/assets/b87c2fd7-2255-41d2-8aca-525536c3d598" />
<img width="568" height="597" alt="Image" src="https://github.com/user-attachments/assets/22409b3e-f74a-434b-9003-fc5060eba470" />
<img width="560" height="612" alt="Image" src="https://github.com/user-attachments/assets/b022685d-84d6-47ce-8f22-61fff7980edc" />
<img width="568" height="600" alt="Image" src="https://github.com/user-attachments/assets/9385a8f6-85b4-4357-94a2-0c3d7e6fdb07" />
