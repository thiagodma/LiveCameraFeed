import cv2
import face_recognition
from pyagender import PyAgender

class FaceAgeGenderDetection(PyAgender):
    #the face detection from PyAgender is not that good so I'll use from other lib
    def detect_faces(self,img,margin=0.2):
        #margin=0.2
        #convert from BGR to RGB
        img = img[:, :, ::-1]
        # import pdb; pdb.set_trace()
        img_h,img_w = img.shape[0],img.shape[1]
        face_locations = face_recognition.face_locations(img)
        face_results = []
        for (top, right, bottom, left) in face_locations:
            x = left
            y = top
            w = right - left
            h = bottom-top
            xi1 = max(int(x - margin * w), 0)
            xi2 = min(int(x + w + margin * w), img_w - 1)
            yi1 = max(int(y - (margin+0.25) * h), 0)
            yi2 = min(int(y + h + margin * h), img_h - 1)
            detection = {'left': xi1, 'top': yi1, 'right': xi2, 'bottom': yi2,
                         'width': (xi2 - xi1), 'height': (yi2 - yi1)}
            face_results.append(detection)

        return face_results

agender = FaceAgeGenderDetection()
img = cv2.imread('data/church.jpeg')
#img = cv2.resize(img, (480, 720))
faces = agender.detect_genders_ages(img)
for face in faces:
#face = faces[0]
    #import pdb; pdb.set_trace()
    cv2.rectangle(img,(face['left'],face['top']),(face['right'],face['bottom']),(0,0,255),2)
    gender = 'Male' if face['gender'] <= 0.5 else 'Female'
    txt = 'Gender:{}, Age:{}'.format(gender,int(face['age']))
    y = face['top']-10 if face['top']>20 else face['bottom']+10
    cv2.putText(img, txt,(face['left'], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

#cv2.rectangle(img,(face['left'],face['right']),(face['bottom'],face['top']),(255,0,0),2)
#cv2.rectangle(img,(face['bottom'],face['top']),(face['left'],face['right']),(255,0,0),2)

cv2.imshow("preview", img)
key = cv2.waitKey()


# img = cv2.imread('data/gabriel.jpeg')
#
# #image = face_recognition.load_image_file("your_file.jpg")
#
# face_locations = face_recognition.face_locations(img)
# #import pdb; pdb.set_trace()
#
# # Display the results
# for top, right, bottom, left in face_locations:
#     # Draw a box around the face
#     cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
#
# # Display the resulting image
# cv2.imshow('Video', img)
#
# cv2.waitKey()
