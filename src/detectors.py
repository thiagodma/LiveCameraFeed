import numpy as np
import tensorflow as tf
import pandas as pd
import cv2, time, socket, struct, warnings
import face_recognition
from pyagender import PyAgender
warnings.filterwarnings('ignore')

class HumanDetector():
    def __init__(self, path_to_ckpt:str,threshold:float=0.7):
        self.path_to_ckpt = path_to_ckpt
        self.threshold = threshold

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.default_graph = self.detection_graph.as_default()
        self.sess = tf.Session(graph=self.detection_graph)

        # Definite input and output Tensors for detection_graph
        self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_humans(self, image):
        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)
        # Actual detection.
        #start_time = time.time()
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})
        #end_time = time.time()

        #print("Elapsed Time:", end_time-start_time)

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()]

    def process_frame(self,boxes,scores,classes,img):
        #calculates the number of persons
        df = pd.DataFrame(zip(scores,classes), columns=['scores','classes'])
        num_people = len(df.loc[(df.classes==1) & (df.scores>self.threshold)])

        #prints the number of persons on the image
        cv2.putText(img, 'Number of people: {}'.format(num_people) ,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        #puts the bounding boxes for the human detection
        for i in range(len(boxes)):
            # Class '1' represents human
            if classes[i] == 1 and scores[i] > self.threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        return num_people, img


    def close(self):
        self.sess.close()
        self.default_graph.close()


class FaceAgeGenderDetection(PyAgender):
    #the face detection from PyAgender is not that good so I'll use face detection from 'face_recognition' module
    def detect_faces(self,img,margin=0.2):
        #convert from BGR to RGB
        img = img[:, :, ::-1]
        # import pdb; pdb.set_trace()
        img_h,img_w = img.shape[0],img.shape[1]
        face_locations = face_recognition.face_locations(img,model='hog',number_of_times_to_upsample=1)
        face_results = []
        for (top, right, bottom, left) in face_locations:
            x,y,w,h = left,top,right-left,bottom-top
            xi1 = max(int(x - margin * w), 0)
            xi2 = min(int(x + w + margin * w), img_w - 1)
            yi1 = max(int(y - (margin+0.25) * h), 0)
            yi2 = min(int(y + h + margin * h), img_h - 1)
            detection = {'left': xi1, 'top': yi1, 'right': xi2, 'bottom': yi2,
                         'width': (xi2 - xi1), 'height': (yi2 - yi1)}
            face_results.append(detection)

        return face_results

    def process_frame(self,faces,img):

        genders = []
        ages = []
        #puts the bounding boxes for the face detection
        for face in faces:
            cv2.rectangle(img,(face['left'],face['top']),(face['right'],face['bottom']),(0,0,255),2)
            gender = 'Male' if face['gender'] <= 0.5 else 'Female'
            txt = 'Gender:{}, Age:{}'.format(gender,int(face['age']))
            y = face['top']-10 if face['top']>20 else face['bottom']+10
            cv2.putText(img, txt,(face['left'], y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,0,255), 1)
            genders.append(gender)
            ages.append(int(face['age']))

        genders = np.array(genders)
        num_males = len(np.where(genders=='Male')[0])
        num_females = len(genders) - num_males

        return ages, num_males, num_females, img
