import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
import custom_utils

from models.fasterrcnn_squeezenet1_1 import create_model

from config import (
    NUM_CLASSES, DEVICE, CLASSES
)

# construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='folder path to input input image (one image or a folder path)',
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
os.makedirs(os.path.join('..\inference_outputs', 'images'), exist_ok=True)

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('..\squeezenet1_1_model\last_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# directory where all the images are present
DIR_TEST = args['input']
test_images = []
if os.path.isdir(DIR_TEST):
    image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
    for file_type in image_file_types:
        test_images.extend(glob.glob(f"{DIR_TEST}/{file_type}"))
else:
    test_images.append(DIR_TEST)
print(f"Test instances: {len(test_images)}")

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = args['threshold']

# to count the total number of frames iterated through
frame_count = 0
# to keep adding the frames' FPS
total_fps = 0
for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    if args['resize']:
        image = cv2.resize(image, (args['resize'], args['resize']))
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float)
    # convert to tensor
    image = torch.tensor(image, dtype=torch.float).cuda()
    # add batch dimension
    image = torch.unsqueeze(image, 0)
    start_time = time.time()
    with torch.no_grad():
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
            orig_image = custom_utils.draw_boxes(orig_image, box, color, args['resize'])
            orig_image = custom_utils.put_class_text(
                orig_image, box, class_name, 
                color, args['resize'] 
            )

        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        cv2.imwrite(f"..\\inference_outputs\\images\\{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)

print('TEST PREDICTIONS COMPLETE')
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")