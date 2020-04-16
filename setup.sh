pip3 install numpy
pip3 install pandas
pip3 install opencv-python
pip3 install tensorflow==1.15.0 #cpu version of tensorflow
pip3 install py-agender[cpu]  # for the cpu version of tensorflow
pip install dash==1.6.1

git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build; cd build; cmake ..; cmake --build .
cd ..
python3 setup.py install

pip3 install face-recognition

wget -O faster_rcnn_inception_v2_coco_2018_01_28.tar.gz  http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

wget -O ssd_mobilenet_v1_coco_2017_11_17.tar.gz http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz

tar -zxvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
tar -zxvf ssd_mobilenet_v1_coco_2017_11_17.tar.gz
