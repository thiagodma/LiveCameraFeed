import numpy as np
import tensorflow as tf
import pandas as pd
import cv2, time, socket, struct, warnings, os, csv, pickle
import face_recognition
from datetime import datetime,timedelta
from pyagender import PyAgender
from detectors import HumanDetector, FaceAgeGenderDetection
warnings.filterwarnings('ignore')

class TrueValue():
    """
    This class applies the human detection and face detection models to get the
    statistics we're interested on.

    Inputs:
        human_detector: an object of the HumanDetector class
        agender_detector: an object of the FaceAgeGenderDetection class
    """
    def __init__(self,human_detector:HumanDetector,agender_detector:FaceAgeGenderDetection):
        self.human_detector = human_detector        #class that does the human detection
        self.agender_detector = agender_detector    #class that does age and gender detection

    def save_stats(self,num_people,ages,num_males,num_females,now):
        """
        This method saves the statistics on a file.

        Inputs:
            num_people: number of people on the image
            ages: list with people's ages
            num_males: number of identified males
            num_females: number of identified females
            now: string with time when the data was collected
        """

        if not os.path.isfile('data.csv'):
            df = pd.DataFrame(columns=['time','num_male','num_female',
                                       'num_people','(0-10)','(10-20)',
                                       '(20-30)','(30-40)','(40-50)',
                                       '(50-60)','(60-70)','(70-inf)'])
            #saves the first datetime available
            with open('first_date.pkl', 'wb') as f: pickle.dump(now, f)
            #saves the basic csv
            df.to_csv('data.csv',index=False)

        #saves the last datetime available
        with open('last_date.pkl', 'wb') as f: pickle.dump(now, f)

        row = [now,num_males,num_females,num_people] + [0]*8
        for age in ages:
            if age <= 10: row[4]+=1
            elif age > 10 and age <=20: row[5]+=1
            elif age > 20 and age <=30: row[6]+=1
            elif age > 30 and age <=40: row[7]+=1
            elif age > 40 and age <=50: row[8]+=1
            elif age > 50 and age <=60: row[9]+=1
            elif age > 60 and age <=70: row[10]+=1
            elif age > 70: row[11]+=1

        with open('data.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(row)


    def run(self,img):

        """
        This method applies both the human detection and face detection models, gets
        the statistics we're interested on and calls the method 'save_stats' to save
        the statistics on a file.

        Inputs:
            img: numpy array that contains the image we want to extract info from

        Outputs:
            img: numpy array that contais the image with all the bounding boxes and
            with the number of people printed.
        """

        #adding 5 hours save in the city's tz
        now = (datetime.now()+timedelta(hours=5)).strftime('%d/%m/%Y %H:%M:%S')

        #detects humans
        boxes, scores, classes = self.human_detector.detect_humans(img)

        #detects genders and ages
        faces = self.agender_detector.detect_genders_ages(img)

        #calculates the number of persons in the image and puts the bounding boxes
        num_people, img = self.human_detector.process_frame(boxes, scores, classes, img)

        #puts the bounding boxes and returns lists with ages and genders
        ages, num_males, num_females, img = self.agender_detector.process_frame(faces,img)

        #saves the stats on a file
        self.save_stats(num_people,ages,num_males,num_females,now)

        return img

if __name__ == '__main__':

    #getting the live feed
    #cap = cv2.VideoCapture()
    #cap.open("http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370")
    model_path = 'faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb'
    #model_path = 'ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb'
    human_detector = HumanDetector(model_path,threshold=0.7)
    agender_detector = FaceAgeGenderDetection()
    truevalue = TrueValue(human_detector,agender_detector)

    #cap = cv2.VideoCapture('data/classroom.mp4')
    cap = cv2.VideoCapture('data/face-demographics-walking.mp4')
    fr=12
    while True:
        for _ in range(fr):
            r, img = cap.read()
        img = cv2.resize(img, (640, 400))
        img = truevalue.run(img)
        #cv2.imwrite('bla.jpg',img)
        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
