import numpy as np
import cv2
from skimage import color

# Function: Poisson blending with Lab color space

def poisson_blending(src, dst, mask):
    # Convert images to Lab color space
    src_lab = color.rgb2lab(src)
    dst_lab = color.rgb2lab(dst)

    # Process the source and destination images
    blended = np.zeros_like(dst_lab)
    for channel in range(3):
        # Poisson blending in each channel
        blended[..., channel] = cv2.seamlessClone(src_lab[..., channel], dst_lab, mask, (src.shape[1]//2, src.shape[0]//2), cv2.NORMAL_CLONE)

    # Convert back to RGB color space
    blended_rgb = color.lab2rgb(blended)
    return blended_rgb

# Test script

def test_poisson_blending():
    # Load images
    src = cv2.imread('source_image.jpg')  # Replace with your source image path
    dst = cv2.imread('destination_image.jpg')  # Replace with your destination image path
    mask = cv2.imread('mask_image.jpg', 0)  # Load the mask image in grayscale

    # Apply Poisson blending
    result = poisson_blending(src, dst, mask)

    # Save the result
    cv2.imwrite('blended_result.jpg', result * 255)  # Save blended image
    print('Blended image saved as blended_result.jpg')

if __name__ == '__main__':
    test_poisson_blending()