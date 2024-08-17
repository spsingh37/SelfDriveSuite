import numpy as np
import cv2
import torch
import os
import time
import argparse
import pathlib
import custom_utils

from models.fasterrcnn_squeezenet1_1 import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', help='path to input video',
    default=r'..\input\inference_data\video_1_trimmed_1.mp4'
)
parser.add_argument(
    '-rs', '--resize', default=None, type=int,
    help='provide an integer to resize the image,\
          e.g. 300 will resize image to 300x300',
)
parser.add_argument(
    '-th', '--threshold', type=float, default=0.25, 
    help='detection threshold to discard detections below this score'
)
args = vars(parser.parse_args())

# For same annotation colors each time.
np.random.seed(42)

# Create inference result dir if not present.
os.makedirs(os.path.join('..\inference_outputs', 'videos'), exist_ok=True)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('..\squeezenet1_1_model\last_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = args['threshold']

cap = cv2.VideoCapture(args['input'])

if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

save_name = str(pathlib.Path(args['input'])).split(os.path.sep)[-1].split('.')[0]
# define codec and create VideoWriter object 
out = cv2.VideoWriter(f"..\\inference_outputs\\videos\\video.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret:
        image = frame.copy()
        if args['resize'] is not None:
            image = cv2.resize(image, (args['resize'], args['resize']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # get predictions for the current frame
            outputs = model(image.to(DEVICE))
        end_time = time.time()
        
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                frame = custom_utils.draw_boxes(frame, box, color, args['resize'])
                frame = custom_utils.put_class_text(
                    frame, box, class_name,
                    color, args['resize']
                )
        cv2.putText(frame, f"{fps:.1f} FPS", 
                    (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 
                    2, lineType=cv2.LINE_AA)

        cv2.imshow('image', frame)
        out.write(frame)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()

# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")