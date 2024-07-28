import os
import numpy as np
from PIL import Image

# Define the directory containing the images
input_dir = r'C:\Users\Lenovo\Downloads\unet_imageseg\dataset_large\val\targets_old'
output_dir = r'C:\Users\Lenovo\Downloads\unet_imageseg\dataset_large\val\targets'

# Ensure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# Mapping of RGB values to category indices
rgb_to_index = {
    (0, 0, 0): 0,         # Undefined
    (210, 0, 200): 1,     # Terrain...pink
    (90, 200, 255): 2,    # Sky...sky blue
    (0, 199, 0): 3,       # Tree...green
    (90, 240, 0): 4,      # Vegetation...light green
    (140, 140, 140): 5,   # Building...grey
    (100, 60, 100): 6,    # Road...dark pinkish
    (250, 100, 255): 7,   # GuardRail...light pink
    (255, 255, 0): 8,     # TrafficSign...yellow
    (200, 200, 0): 9,     # TrafficLight...brownish yellow
    (255, 130, 0): 10,    # Pole...orange
    (80, 80, 80): 11,     # Misc...dark grey
    (160, 60, 60): 12,    # Truck...brown red
    (255, 127, 80): 13,   # Car...light orange
    (0, 139, 139): 14     # Van...bluish
}

# Function to convert RGB image to segmentation mask
def convert_to_mask(image_path, save_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    
    # Initialize the mask with zeros
    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
    
    # Iterate over the RGB to index mapping and update the mask
    for rgb, index in rgb_to_index.items():
        # Create a mask for the current RGB value
        match = np.all(image_np == rgb, axis=-1)
        mask[match] = index
    
    # Save the mask as a PNG image
    mask_image = Image.fromarray(mask)
    mask_image.save(save_path)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        convert_to_mask(input_path, output_path)
        print(f'Converted {filename} to segmentation mask.')

print('Conversion completed.')
