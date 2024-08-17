import albumentations as A
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from config import DEVICE, CLASSES

plt.style.use('ggplot')

# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))

# define the training tranforms
def get_train_transform():
    return A.Compose([
        A.MotionBlur(blur_limit=3, p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, p=0.5
        ),
        A.ColorJitter(p=0.5),
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc',
        'label_fields': ['labels']
    })

# define the validation transforms
def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0),
    ], bbox_params={
        'format': 'pascal_voc', 
        'label_fields': ['labels']
    })


def show_tranformed_image(train_loader):
    """
    This function shows the transformed images from the `train_loader`.
    Helps to check whether the tranformed images along with the corresponding
    labels are correct or not.
    Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
    """
    if len(train_loader) > 0:
        for i in range(3):
            images, targets = next(iter(train_loader))
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
            labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
            sample = images[i].permute(1, 2, 0).cpu().numpy()
            sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
            for box_num, box in enumerate(boxes):
                cv2.rectangle(sample,
                            (box[0], box[1]),
                            (box[2], box[3]),
                            (0, 0, 255), 2)
                cv2.putText(sample, CLASSES[labels[box_num]], 
                            (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 255), 2)
            cv2.imshow(
                'Transformed image', 
                sample
            )
            cv2.waitKey(0)
            cv2.destroyAllWindows()

def save_model(save_dir, epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenever called.

    :param epoch: The epoch number.
    :param model: The neural network model.
    :param optimizer: The optimizer.
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, f"{save_dir}/last_model.pth")

def save_train_loss_plot(save_dir, train_loss_list):
    """
    Function to save both train loss graph.
    
    :param save_dir: Path to save the graphs.
    :param train_loss_list: List containing the training loss values.
    """
    figure_1, train_ax = plt.subplots()
    train_ax.plot(train_loss_list, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    figure_1.savefig(f"{save_dir}/train_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')

def draw_boxes(image, box, color, resize=None):
    """
    This function will annotate images with bounding boxes 
    based on wether resizing was applied to the image or not.

    :param image: Image to annotate.
    :param box: Bounding boxes list.
    :param color: Color to apply to the bounding box.
    :param resize: Either None, or provide a single integer value,
                   if 300, image will be resized to 300x300 and so on.

    Returns:
           image: The annotate image.
    """
    if resize is not None:
        cv2.rectangle(image,
                    (
                        int((box[0]/resize)*image.shape[1]), 
                        int((box[1]/resize)*image.shape[0])
                    ),
                    (
                        int((box[2]/resize)*image.shape[1]), 
                        int((box[3]/resize)*image.shape[0])
                    ),
                    color, 2)
        return image
    else:
        cv2.rectangle(image,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color, 2)
        return image

def put_class_text(image, box, class_name, color, resize=None):
    """
    Annotate the image with class name text.

    :param image: The image to annotate.
    :param box: List containing bounding box coordinates.
    :param class_name: Text to put on bounding box.
    :param color: Color to apply to the text.
    :param resize: Whether annotate according to resized coordinates or not.

    Returns:
           image: The annotated image.
    """
    if resize is not None:
        cv2.putText(image, class_name, 
                    (
                        int(box[0]/resize*image.shape[1]), 
                        int(box[1]/resize*image.shape[0]-5)
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        return image
    else:
        cv2.putText(image, class_name, 
                    (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                    2, lineType=cv2.LINE_AA)
        return image