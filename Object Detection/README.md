## 🤖 Traffic Sign Detection
For traffic sign detection, I have used Faster-RCNN with Squeezenet1_1 backbone (for achieving real-time results)

# Data download
Downloading the data is not necessary if you've cloned this repo, because the data is ready for training/testing. Nonetheless, if you're still interested then first download the GTSD (German Traffic Sign Detection) dataset, specifically you need the following 2 files placed in the input directory:<br>
- TrainIJCNN2013.zip
- TestIJCNN2013.zip 

Post-download and extracting the above directories, the input directory structure should like this:
```
├── input
│   ├── inference_data
│   │   └── video_1_trimmed_1.mp4
│   ├── TestIJCNN2013
│   │   └── TestIJCNN2013Download
│   ├── TrainIJCNN2013
│   ├── classes_list.txt
│   ├── gt.txt
│   ├── MY_README.txt
│   ├── signnames.csv
```

# Data preprocessing & train/val split
Carry out the following necessary steps though again not necessary unless you want to change the train/val ratio
```
cd src
python txt_to_csv.py
python split_train_valid.py
python csv_to_xml.py
```

Post-dprocessing, the input directory structure should like this:
```
├── input
│   ├── inference_data
│   │   └── video_1_trimmed_1.mp4
│   ├── TestIJCNN2013
│   │   └── TestIJCNN2013Download
│   ├── TrainIJCNN2013
│   ├── train_images 
│   ├── train_xmls 
│   ├── valid_images 
│   ├── valid_xmls 
│   ├── all_annots.csv
│   ├── classes_list.txt
│   ├── gt.txt
│   ├── MY_README.txt
│   ├── signnames.csv
│   ├── train.csv
│   └── valid.csv
```

# Train
```
python train.py
```

# Test/Demo
First download the pre-trained weights from <a href="https://drive.google.com/file/d/1jtEyGtba3uN0VVWfucvD5ap3JKDy-YHr/view?usp=sharing">here</a> and place it in the squeezenet1_1_model directory.<br>
When testing on a directory of images
```
python inference.py -i /path/to/directory
```

When testing on a video file
```
python inference_video.py -i /path/to/video/file
```
