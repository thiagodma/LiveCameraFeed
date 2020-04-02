pip3 install numpy==1.15.4
pip3 install pandas==0.25.0
pip3 install cv2==3.4.9
pip3 install tensorflow==1.15 #cpu version of tensorflow
pip3 install py-agender[cpu]  # for the cpu version of tensorflow

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install

pip3 install face_recognition

wget -O faster_rcnn_inception_v2_coco_2018_01_28.tar.gz  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

tar -zxvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

mkdir graphs
mkdir proc_imgs
