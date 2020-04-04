import numpy as np
import tensorflow as tf
import pandas as pd
import cv2, time, socket, struct, warnings
import face_recognition
from pyagender import PyAgender
warnings.filterwarnings('ignore')

class HumanDetector():
    """
    This class uses a trained model from the tensorflow's object detection model zoo
    to detect people.

    Inputs:
        path_to_ckpt: the path to the model for inference
        threshold: parameter that sets the minimum model's confidence to say the object is
        a person. Its a float in range (0,1)
    """
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
        self.detection_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    def detect_humans(self, image):
        """
        This method does the human detection.

        Inputs:
            image: is a numpy array that contains the image

        Outputs:
            boxes: list of tuples with keys to the bounding box of each detected object
            scores: list with the confidences for each detection
            classes: list with the class of each detection
        """

        # Expand dimensions since the trained_model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image, axis=0)

        # Actual detection.
        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        im_height, im_width,_ = image.shape
        boxes_list = [None for i in range(boxes.shape[1])]
        for i in range(boxes.shape[1]):
            boxes_list[i] = (int(boxes[0,i,0] * im_height),
                        int(boxes[0,i,1]*im_width),
                        int(boxes[0,i,2] * im_height),
                        int(boxes[0,i,3]*im_width))

        return boxes_list, scores[0].tolist(), [int(x) for x in classes[0].tolist()]

    def process_frame(self,boxes,scores,classes,img):
        """
        This method prints the number of people and correspondent bounding boxes
        on the input image and returns the number of people on the image.

        Inputs:
            boxes: list of tuples with keys to the bounding box of each detected object
            scores: list with the confidences for each detection
            classes: list with the class of each detection
            img: numpy array the contais the image that will be printed

        Outputs:
            num_people: number of people on the image
            img: img with the bounding boxes and number of people printed
        """

        #calculates the number of persons (class '1' means person)
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
    """
    This class uses two modules to detect faces and to predict gender and age.
    The module that predicts age and gender is 'pyagender' and the module that
    finds faces is 'face_detection'. (pyagender's face detection was not good enough)

    Inputs:
        model: is a string and can be either 'hog' or 'cnn'. 'cnn' is more accurate but slower.
        up_times: integer in range (1,inf) that specifies the number of times to upsample the image.
        Larger values allow better detection of little faces.
    """

    def __init__(self,model:str='hog',up_times:int=1):
        super().__init__()
        self.model = model
        self.up_times = up_times

    def detect_faces(self,img,margin=0.2):
        """
        This method does the face detection.

        Inputs:
            image: is a numpy array that contains the image
            margin: is a float that determines the size of the margin on the face detection.

        Outputs:
            faces: list of dictionaries. Each dictionary has the fields: 'left',
            'top','right','bottom','width' and 'height' which are the coordinates to the
            bounding boxes.
        """
        #converts from BGR to RGB
        img = img[:, :, ::-1]
        img_h,img_w = img.shape[0],img.shape[1]
        face_locations = face_recognition.face_locations(img,model=self.model,number_of_times_to_upsample=self.up_times)
        faces = []
        for (top, right, bottom, left) in face_locations:
            x,y,w,h = left,top,right-left,bottom-top
            xi1 = max(int(x - margin * w), 0)
            xi2 = min(int(x + w + margin * w), img_w - 1)
            yi1 = max(int(y - (margin+0.25) * h), 0)
            yi2 = min(int(y + h + margin * h), img_h - 1)
            detection = {'left': xi1, 'top': yi1, 'right': xi2, 'bottom': yi2,
                         'width': (xi2 - xi1), 'height': (yi2 - yi1)}
            faces.append(detection)

        return faces

    def process_frame(self,faces,img):
        """
        This method prints the bounding boxes on the input image and returns
        some statistics.

        Inputs:
            faces: list of dictionaries. Each dictionary has the fields: 'left',
            'top','right','bottom','width' and 'height' which are the coordinates to the
            bounding boxes.
            img: numpy array the contais the image that will be printed

        Outputs:
            ages: list with people's ages
            num_males: number of identified males
            num_females: number of identified females
            img: img with the bounding boxes printed
        """

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
