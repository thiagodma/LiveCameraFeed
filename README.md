# Live Camera Feed

## Introduction

This repo contains the code used to do an application that does:
* [Human detection](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
* [Face detection](https://github.com/ageitgey/face_recognition)
* [Gender and Age estimation](https://github.com/aristofun/py-agender)

The application is fed with a live camera feed and makes plots with statistics obtained from the video.

<img src="https://raw.githubusercontent.com/thiagodma/LiveCameraFeed/master/images/img.jpg" width="500" height="500" align="center" />

#### Folder and File Description
* **images/** : folder with images for the readme file
* **src/** : folder with the source code
* **src/main.py** : file that generates the web app
* **src/detectors.py** : file with the classes for object detection and face detection (plus gender and age estimation)
* **src/true_value.py** : file with the class that uses the detectors
* **setup.sh** : setup file that downloads python packages and downloads the models for object detection.

## Setup
To run this code in your computer I suggest you to create a virtual environment. With you virtual environment activated, clone this repo. To install all the packages and download the important files just run:
~~~
./setup.sh
~~~

With your environment set up you can run the app by running:
~~~
python src/main.py
~~~

If you want to run only the human detection/face detection plus age and gender estimation (without plotting the stats), run:
~~~
python src/true_value.py
~~~


## Limitations

The main limitation is related to computational resources. The app runs in a server that doesn't have a GPU and doesn't have much RAM available.

The app uses the HOG algorithm for face detection and can upsample the image only once (thus it struggles to detect faces of people that are far from the camera). If running on a server with higher computational resources, is easy to change the face detection algorithm to a CNN and let the algorithm upsample the image more times.

Due to the camera placement on the laundry, the human detection model doesn't detect people that are sat on the chairs.

## Scale

The code was written in such way that the application is easily scalabe.

* Is possible to change the models, so that one can fit the app in any server
* Is easy to change the code so that it can track multiple video sources
* The app is hosted on a cloud server and is easy to change the computational resourcers available
* For the sake of simplicity, the app stores the gathered data in a csv file. In order to scale this app, is important to use a proper database

## Risks

In order to successfully deploy an application that uses people's images it is important to try to find all the risks related to the application. Here I'll point out some of them:

* It is important to be very clear with what data the app is collecting: does it collect sensible data? does the app saves people's faces? is it really only collecting the number of people in a place?
* As it is really easy to change the video feed on this app is important to make sure it won't get in the hands of people with malicious intentions. Ex: what if a dictatorial government uses this technology to keep track of rebels?
* It is really important to make sure that all the data gathered is well secured, for people with malicious intentions might try to use it.
* As the data available for training these machine learning models usually come from the US and Europe, there's a considerable risk that the models are biased for caucasian people.
