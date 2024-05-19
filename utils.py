import numpy as np
import cv2
import matplotlib.pyplot as plt
from bm3d import bm3d

def apply_bm3d_denoiser(rgb_image, sigma):
    """
    Apply BM3D denoiser to an RGB image using the bm3d library.
    
    Args:
    - rgb_image: Input RGB image (numpy array).
    - sigma: Standard deviation of the noise in the image.
    
    Returns:
    - Denoised RGB image (numpy array).
    """
    # Apply BM3D denoiser to the RGB image
    denoised_image = bm3d(rgb_image, sigma)
    
    return denoised_image

def resize_image(image, target_height=237, target_width=374):
    """
    Resize an image to the specified dimensions while handling padding for shrinking or zooming.
    
    Args:
    - image: Input image as a numpy array.
    - target_height: Target height for resizing.
    - target_width: Target width for resizing.
    
    Returns:
    - Resized image as a numpy array with dimensions (target_height, target_width, 3).
    """
    # Get the original dimensions of the image
    original_height, original_width, _ = image.shape
    
    # Calculate scaling factors for resizing
    scale_height = target_height / original_height
    scale_width = target_width / original_width
    
    # Decide whether to shrink or zoom the image based on scaling factors
    if scale_height < 1 or scale_width < 1:
        # Shrink the image by adding padding
        new_height = min(target_height, original_height)
        new_width = min(target_width, original_width)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Add padding to match the target dimensions
        top_padding = (target_height - new_height) // 2
        bottom_padding = target_height - new_height - top_padding
        left_padding = (target_width - new_width) // 2
        right_padding = target_width - new_width - left_padding
        resized_image = cv2.copyMakeBorder(resized_image, top_padding, bottom_padding, left_padding, right_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        # Zoom the image by adding padding
        new_height = max(target_height, original_height)
        new_width = max(target_width, original_width)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Crop the image to match the target dimensions
        start_x = (new_width - target_width) // 2
        start_y = (new_height - target_height) // 2
        resized_image = resized_image[start_y:start_y + target_height, start_x:start_x + target_width]
    
    return resized_image

# Example usage:
# Load an image using OpenCV
# img = cv2.imread("image.jpg")
# Resize the image
# resized_img = resize_image(img, target_height=237, target_width=374)


def restore_image(image, original_height, original_width):
    """
    Restore an image to its original dimensions from a resized image.
    
    Args:
    - image: Resized image as a numpy array.
    - original_height: Original height of the image.
    - original_width: Original width of the image.
    
    Returns:
    - Restored image as a numpy array with dimensions (original_height, original_width, 3).
    """
    # Get the dimensions of the resized image
    resized_height, resized_width, _ = image.shape
    
    # Calculate the scaling factors for zooming or shrinking
    scale_height = original_height / resized_height
    scale_width = original_width / resized_width
    
    # Decide whether to zoom or shrink the image based on scaling factors
    if scale_height < 1 or scale_width < 1:
        # Zoom the image by merging pixels
        restored_image = cv2.resize(image, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
    else:
        # Shrink the image by adding new pixels
        restored_image = cv2.resize(image, (original_width, original_height), interpolation=cv2.INTER_AREA)
    
    return restored_image

# Example usage:
# original_height, original_width = 1707, 2560
# restored_img = restore_image(resized_img, original_height, original_width)



