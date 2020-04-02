import numpy as np
import tensorflow as tf
import pandas as pd
import cv2, time, socket, struct, warnings
from pyagender import PyAgender
warnings.filterwarnings("ignore")

class DetectorAPI:
    def __init__(self, path_to_ckpt):
        self.path_to_ckpt = path_to_ckpt

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

    def processFrame(self, image):
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

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()], int(num[0])

    def close(self):
        self.sess.close()
        self.default_graph.close()

if __name__ == "__main__":

    #object that does human detection
    odapi = DetectorAPI(path_to_ckpt='faster_rcnn_inception_v2_coco_2018_01_28/frozen_inference_graph.pb')
    threshold = 0.7

    #object that does the age and gender classification
    agender = PyAgender()

    #getting the live feed
    #cap = cv2.VideoCapture()
    #cap.open("http://81.14.37.24:8080/mjpg/video.mjpg?timestamp=1585844515370")
    cap = cv2.VideoCapture('data/face-demographics-walking.mp4')

    while True:
        r, img = cap.read()
        #import pdb; pdb.set_trace()
        #img = cv2.resize(img, (1280, 720))

        #detects humans
        boxes, scores, classes, num = odapi.processFrame(img)

        #detects faces then genders and ages
        faces = agender.detect_genders_ages(img)

        #calculates the number of persons
        df = pd.DataFrame(zip(scores,classes), columns=['scores','classes'])
        num_persons = len(df.loc[(df.classes==1) & (df.scores>threshold)])

        #prints the number of persons
        cv2.putText(img, 'Number of persons: {}'.format(num_persons) ,(0,40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

        #puts the bounding boxes for the human detection
        for i in range(len(boxes)):
            # Class '1' represents human
            if classes[i] == 1 and scores[i] > threshold:
                box = boxes[i]
                cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)

        for face in faces:
            cv2.rectangle(img,(face['left'],face['top']),(face['right'],face['bottom']),(0,0,255),2)
            gender = 'Male' if face['gender'] <= 0.5 else 'Female'
            txt = 'Gender:{}, Age:{}'.format(gender,int(face['age']))
            y = face['top']-10 if face['top']>20 else face['bottom']+10
            cv2.putText(img, txt,(face['left'], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

        cv2.imshow("preview", img)
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
