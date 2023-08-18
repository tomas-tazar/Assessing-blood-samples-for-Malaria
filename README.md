# Assessing blood samples for malaria

<p align="center">
  <img src="https://github.com/tomas-tazar/Assessing-blood-samples-for-Malaria/blob/main/Documentation/Open%20Day%20Presentation%20Images/readmepipeline.png" width="450" /> 
  <img src="https://github.com/tomas-tazar/Assessing-blood-samples-for-Malaria/blob/main/Documentation/Open%20Day%20Presentation%20Images/readmeimg.jpg" width="400" />
</p>

## Introduction

The purpose of this project was to develop a software solution capable of **classifying**,**segmenting** and **evaluating** red blood cells within at-risk patients. Combining the power of machine learning-based classification and computer vision-based segmentation, we provide an end-to-end solution which is able to successfully identify parasitized and uninfected cells within a 
blood smear image. The system provides a **high level of accuracy & precision** in its predictions, which are then used to **diagnose** and **treat** malaria patients.

## Classification

This project utilizes Keras (**Sequential** & **Sequential with Spatial Attention**) and **state-of-the-art VGG16** (**Large-Scale Image Recognition Convolutional Network**) models which have been trained on a database of parasitized and uninfected NIH (National Institute of Health) cells. The dataset has been pre-processed into appropriate compatibility for model training. 

The models were all able to reach a high level of training accuracy, the **Sequential model** scored the highest training accuracy of **0.95** and an prediction f-score of **1**. All models were able to reach a high precision, recall and f-score respectively. These models were then saved for later use within the end-to-end system, where they provide infection status predictions of segmented contours. 

The saved models can be found within the '*Dependencies/*' folder of the project under '*model_information/models/*'. 

## Segmentation

Segmentation techniques such as **Otsu's tresholding** and **K-means clustering** were employed in our system which allowed us to isolate individual cells 
from the UCL (University College London) blood smear dataset. Techniques such as Gaussian blurring were carried out to de-noise the image before isolating. 
Thresholding and contouring was applied to obtain extracted red blood cells from the image which were then filtered for size. 

Bounding rectangles were then drawn around the contours which marked our **ROIs** (Regions of Interest). **ROIs** were then passed to the classification models for
infection status prediction. Ultimately, the segmented cells were marked with a red, green or blue bounding box depending on the infection status.

## Running the end-to-end system

The software solution is provided as a command line application which can be run from the command line with the following command: ` python3 malaria_evaluation.py `.
This contains all the necessary information for the user to run the end-to-end system.

The program will prompt the user to choose which segmentation method they would like to use, following that, the user is able to choose from a list of trained models as
mentioned before. The program will then segment and evaluate the images or folders that is provided by the user, or use the default evaluation setting.

An interactive view of the evaluated images will be displayed unless the user chooses only to save the images to a folder or save the image information.

## Dependencies 

In order for the end-to-end system to run correctly, a **Python version of 3.10 or higher is recommended**.

The following libraries are required to run the end-to-end system:
- numpy, keras, tensorflow, matplotlib, opencv-python, scikit-learn

Classification models are required for the program to run, tthese are available to download at the following link: [Models](https://drive.google.com/drive/folders/1pBxND8jQNw32ySYwk9Q0NtPctUS66t9u?usp=sharing)
This folder needs to be included in the "Dependencies" folder.

It is highly recommended to use a virtual environment to avoid any conflicts with other projects, the following [anaconda](https://www.anaconda.com/products/distribution) enviroment
is recommended to be instantiated: ` conda create -n malaria python=3.10 `.

After this is done, we can activate the environment with the following command: ` conda activate malaria ` and install the required libraries.

We are able to install these libraries with the following command: ` pip3 install {library_name} `. We are now ready to run the end-to-end system.

## Datasets 

Datasets of images have been provided by the NIH (National Institute of Health) and by UCL (University College London) for training, testing and validation.
External testing datasets such as MNIST (Modified National Institute of Standards and Technology) and CIL (The Cell Image Library) have also been used for testing purposes.

The dataset required for default segmentation can be found downloaded at the following link: [malaria-UCL](http://vase.essex.ac.uk/xfer/malaria-ucl.zip).
This dataset will need to be inserted into the '*Dependencies/*' folder of the project for default evaluation to function properly.

The dataset required for training the classification models can be found at the following link: [malaria-NIH](http://vase.essex.ac.uk/xfer/malaria-nih.zip).

## Future work

- [ ] Porting the software to a mobile application to enable evaluation on the go.
- [ ] Scaling the software to improve its adaptation to large-scale data processing.
- [ ] Diversifying choice of segmentation and classification methods.
- [ ] Adapting the classification to a wider range of light conditions.
