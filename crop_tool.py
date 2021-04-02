import cv2
import numpy as np

def crop_and_resize(image, person_box, size):
    y1, x1, y2, x2 = person_box
    im_h, im_w, _ = image.shape
    pad_x1, pad_y1, pad_x2, pad_y2 = 0, 0, 0, 0
    
    if y1 < 0:
        pad_y1 = abs(y1)
        y1 = 0
    if y2 > im_h:
        pad_y2 = y2 - im_h 
    if x1 < 0:
        pad_x1 = abs(x1)
        x1 = 0
    if x2 > im_h:
        pad_x2 = x2 - im_w

    
    if not np.equal([pad_x1, pad_y1, pad_x2, pad_y2], 0).all():
        image = cv2.copyMakeBorder(image, pad_y1, pad_y2, pad_x1, pad_x2, cv2.BORDER_CONSTANT)
    
    image = image[y1:y2, x1:x2]
    image = cv2.resize(image, (size, size))
    return image
    