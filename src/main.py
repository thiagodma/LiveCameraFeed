# import numpy as np
# import tensorflow as tf
# import pandas as pd
# import cv2, time, socket, struct, warnings
# import face_recognition
# from pyagender import PyAgender
# from detectors import HumanDetector, FaceAgeGenderDetection
# from truevalue import TrueValue
# warnings.filterwarnings('ignore')
#
# model_path = '../faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
# human_detector = HumanDetector(model_path)
#
# agender_detector = FaceAgeGenderDetection()
#
# video_link = 'http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370'
# truevalue = TrueValue(human_detector,agender_detector,video_link,'../proc_imgs','../graphs')
# truevalue.run()


import dash
import dash_core_components as dcc
import dash_html_components as html
from true_value import TrueValue
from detectors import HumanDetector, FaceAgeGenderDetection
from flask import Flask, Response
import cv2
# import pdb; pdb.set_trace()
model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
human_detector = HumanDetector(model_path)
agender_detector = FaceAgeGenderDetection()
truevalue = TrueValue(human_detector,agender_detector)

class VideoCamera():
    def __init__(self):
        cap = cv2.VideoCapture()
        cap.open('http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370')
        #cap = cv2.VideoCapture('data/face-demographics-walking.mp4')
        self.video = cap

    def __del__(self):
        self.video.release()

    def get_frame(self):
        _ , img = self.video.read()
        img = truevalue.run(img)
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield(b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div([
    html.H1("Webcam Test"),
    html.Img(src="/video_feed")
])

if __name__ == '__main__':
    app.run_server(debug=True)



#getting the live feed
# cap = cv2.VideoCapture()
# cap.open(self.video_link)
#
# while True:
#     _ , img = cap.read()
