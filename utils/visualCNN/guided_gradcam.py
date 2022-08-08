"""
Created on Thu Oct 23 11:27:15 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from utils.visualCNN.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from utils.visualCNN.gradcam import GradCam
from utils.visualCNN.guided_backprop import GuidedBackprop


def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


def execute(model, layerID, inputs, classID, file_name_to_export):
    try:
        # Grad cam
        gcv2 = GradCam(model, target_layer=layerID)
        # Generate cam mask
        cam = gcv2.generate_cam(inputs, classID)
        print('Grad cam completed')

        # Guided backprop
        GBP = GuidedBackprop(model)
        # Get gradients
        guided_grads = GBP.generate_gradients(inputs, classID)
        print('Guided backpropagation completed')

        # Guided Grad cam
        cam_gb = guided_grad_cam(cam, guided_grads)
        save_gradient_images(cam_gb, file_name_to_export + '_GGrad_Cam')
        grayscale_cam_gb = convert_to_grayscale(cam_gb)
        save_gradient_images(grayscale_cam_gb, file_name_to_export + '_GGrad_Cam_gray')
        print('Guided grad cam completed')
    except Exception as e:
        print("Couldn't execute: 'Guided grad cam "+str(e))
