# Live Camera Feed

## Introduction

This repo contains the code used to do an application that does:
* Human detection
* Face detection
* Gender and Age estimation

The application is fed with a live camera feed and makes plots with statistics obtained from the video.

<img src="https://raw.githubusercontent.com/thiagodma/LiveCameraFeed/master/images/img.jpg" width="500" height="500" align="center" />

#### Folder and File Description
* images/ : folder with images for the readme file
* setup.sh : setup file that downloads python packages and downloads the models for object detection.
* src/main.py : file that generates the web app
* src/detectors.py : file with the classes for object detection and face detection (plus gender and age estimation)
* src/true_value.py : file with the class that uses the detectors

## Limitations

The main limitation is related to computational resources. The app runs in a server that doesn't have a GPU and doesn't have much RAM available.

The app uses the HOG algorithm for face detection and can upsample the image only once (thus it struggles to detect faces of people that are far from the camera). If running on a server with higher computational resources, is easy to change the face detection algorithm to a CNN and let the algorithm upsample the image more times.

Due to the camera placement on the laundry, the human detection model doesn't detect people that are sat on the chairs.

## Scale

The code was written in such way that the application is easily scalabe.

* Is possible to change the models, so that one can fit the app in any server
* Is easy to change the code so that it can track multiple video sources
* The app is hosted on a cloud server and is easy to change the computational resourcers available

## Risks
