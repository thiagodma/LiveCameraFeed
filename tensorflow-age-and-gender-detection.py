from pyagender import PyAgender
import cv2
import warnings

warnings.filterwarnings("ignore")
agender = PyAgender()

#cap=cv2.VideoCapture(0)


#while True:
#r, img = cap.read()
img = cv2.imread('data/celebs.jpeg')
#img = cv2.resize(img, (720, 480))
faces = agender.detect_genders_ages(img)
for face in faces:
#face = faces[0]
    import pdb; pdb.set_trace()
    cv2.rectangle(img,(face['left'],face['top']),(face['right'],face['bottom']),(0,0,255),2)
    gender = 'Male' if face['gender'] <= 0.5 else 'Female'
    txt = 'Gender:{}, Age:{}'.format(gender,int(face['age']))
    y = face['top']-10 if face['top']>20 else face['bottom']+10
    cv2.putText(img, txt,(face['left'], y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

#cv2.rectangle(img,(face['left'],face['right']),(face['bottom'],face['top']),(255,0,0),2)
#cv2.rectangle(img,(face['bottom'],face['top']),(face['left'],face['right']),(255,0,0),2)

cv2.imshow("preview", img)
key = cv2.waitKey()
#import pdb;pdb.set_trace()
print(faces)
#img = cv2.resize(img, (1280, 720))

#boxes, scores, classes, num = odapi.processFrame(img)
#df = pd.DataFrame(zip(scores,classes), columns=['scores','classes'])
#print('Number of persons: {}'.format(len(df.loc[(df.classes==1) & (df.scores>threshold)])))
#import pdb;pdb.set_trace()

# Visualization of the results of a detection.

# for i in range(len(boxes)):
#     # Class 1 represents human
#     if classes[i] == 1 and scores[i] > threshold:
#         box = boxes[i]
#         cv2.rectangle(img,(box[1],box[0]),(box[3],box[2]),(255,0,0),2)
#
# cv2.imshow("preview", img)
# key = cv2.waitKey(1)
# if key & 0xFF == ord('q'):
#     break
