import numpy as np
import tensorflow as tf
import pandas as pd
import cv2, time, socket, struct, warnings, os, csv
import face_recognition
from datetime import datetime
from pyagender import PyAgender
from detectors import HumanDetector, FaceAgeGenderDetection
warnings.filterwarnings('ignore')

class TrueValue():
    def __init__(self,human_detector:HumanDetector,agender_detector:FaceAgeGenderDetection):
        self.human_detector = human_detector        #class that does the human detection
        self.agender_detector = agender_detector    #class that does age and gender detection

    def save_stats(self,num_persons,ages,num_males,num_females,now):
        if not os.path.isfile('../data.csv'):
            df = pd.DataFrame(columns=['time','num_male','num_female',
                                       'num_persons','(0-10)','(10-20)',
                                       '(20-30)','(30-40)','(40-50)',
                                       '(50-60)','(60-70)','(70-inf)'])
            df.to_csv('../data.csv',index=False)

        row = [now,num_males,num_females,num_persons] + [0]*8
        for age in ages:
            if age <= 10: row[4]+=1
            elif age > 10 and age <=20: row[5]+=1
            elif age > 20 and age <=30: row[6]+=1
            elif age > 30 and age <=40: row[7]+=1
            elif age > 40 and age <=50: row[8]+=1
            elif age > 50 and age <=60: row[9]+=1
            elif age > 60 and age <=70: row[10]+=1
            elif age > 70: row[11]+=1

        with open('../data.csv','a') as f:
            writer = csv.writer(f)
            writer.writerow(row)


    def run(self,img):

        now = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        all_nows.append(now)

        #detects humans
        boxes, scores, classes = self.human_detector.detect_humans(img)

        #detects genders and ages
        faces = self.agender_detector.detect_genders_ages(img)

        #calculates the number of persons in the image and puts the bounding boxes
        num_persons, img = self.human_detector.process_frame(boxes, scores, classes, img)
        all_num_persons.append(num_persons)

        #puts the bounding boxes and returns lists with ages and genders
        ages, num_males, num_females, img = self.agender_detector.process_frame(faces,img)

        #saves the processed frame as an image
        #cv2.imwrite(self.proc_img_path + '/' + now + '.jpg',img)

        #saves the stats
        self.save_stats(num_persons,ages,num_males,num_females,now)

        return img
