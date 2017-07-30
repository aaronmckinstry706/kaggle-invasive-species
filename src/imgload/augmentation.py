import math
import random

import scipy.misc as misc

def resize_and_random_crop(desired_width):
    """Returns a cropping function, F. F takes an image I of shape (w,h,3) where
    w is width, h is height, and 3 is the number of color channels in the image.
    F returns a random desired_width-by-desired_width crop of I. 
    
    Arguments:
        image -- A numpy array. 
    
    Returns:
        F -- A random image cropping function. 
    """
    def function(image):
        if image.shape[0] <= image.shape[1]:
            min_dim = 0
        else:
            min_dim = 1
        
        scale_factor = float(desired_width)/image.shape[min_dim]
        new_dimensions = [
            int(round(image.shape[0]*scale_factor)),
            int(round(image.shape[1]*scale_factor)),
            3]
        new_dimensions[min_dim] = desired_width
        image = misc.imresize(image, size=tuple(new_dimensions))
        
        max_length = max(new_dimensions[0], new_dimensions[1])
        start_index = random.choice(range(0, max_length - desired_width + 1))
        
        if min_dim == 0:
            return image[:,start_index:start_index+desired_width,:]
        else:
            return image[start_index:start_index+desired_width,:,:]
    
    return function

def resize_and_center_crop(desired_width):
    """Returns a cropping function, F. F takes an image I of shape (w,h,3) where
    w is width, h is height, and 3 is the number of color channels in the image.
    F returns a centered desired_width-by-desired_width crop of I. 
    
    Arguments:
        image -- A numpy array. 
    
    Returns:
        F -- A center image cropping function. 
    """
    def function(image):
        if image.shape[0] <= image.shape[1]:
            min_dim = 0
        else:
            min_dim = 1
        
        scale_factor = float(desired_width)/image.shape[min_dim]
        new_dimensions = [
            int(round(image.shape[0]*scale_factor)),
            int(round(image.shape[1]*scale_factor)),
            3]
        new_dimensions[min_dim] = desired_width
        image = misc.imresize(image, size=tuple(new_dimensions))
        
        max_length = max(new_dimensions[0], new_dimensions[1])
        start_index = int(math.floor((max_length - desired_width)/2.0))
        
        if min_dim == 0:
            return image[:,start_index:start_index+desired_width,:]
        else:
            return image[start_index:start_index+desired_width,:,:]
    
    return function
