"""
Created on Wed Mar 28 10:12:13 2018

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import numpy as np

from torch.autograd import Variable
import torch

from utils.visualCNN.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from utils.visualCNN.vanilla_backprop import VanillaBackprop
# from guided_backprop import GuidedBackprop  # To use with guided backprop


def generate_smooth_grad(Backprop, prep_img, target_class, param_n, param_sigma_multiplier):
    """
        Generates smooth gradients of given Backprop type. You can use this with both vanilla
        and guided backprop
    Args:
        Backprop (class): Backprop type
        prep_img (torch Variable): preprocessed image
        target_class (int): target class of imagenet
        param_n (int): Amount of images used to smooth gradient
        param_sigma_multiplier (int): Sigma multiplier when calculating std of noise
    """
    # Generate an empty image/matrix
    smooth_grad = np.zeros(prep_img.size()[1:])

    mean = 0
    sigma = param_sigma_multiplier / (torch.max(prep_img) - torch.min(prep_img)).item()
    for x in range(param_n):
        # Generate noise
        noise = Variable(prep_img.data.new(prep_img.size()).normal_(mean, sigma**2))
        # Add noise to the image
        noisy_img = prep_img + noise
        # Calculate gradients
        vanilla_grads = Backprop.generate_gradients(noisy_img, target_class)
        # Add gradients to smooth_grad
        smooth_grad = smooth_grad + vanilla_grads
    # Average it out
    smooth_grad = smooth_grad / param_n
    return smooth_grad


def execute(model, inputs, classID, file_name_to_export, param_n = 50,param_sigma_multiplier = 4):
    try:
        VBP = VanillaBackprop(model)
        # GBP = GuidedBackprop(pretrained_model)  # if you want to use GBP dont forget to
        # change the parametre in generate_smooth_grad
        
        smooth_grad = generate_smooth_grad(VBP,  # ^This parameter
                                        inputs,
                                        classID,
                                        param_n,
                                        param_sigma_multiplier)

        # Save colored gradients
        save_gradient_images(smooth_grad, file_name_to_export + '_SmoothGrad_color')
        # Convert to grayscale
        grayscale_smooth_grad = convert_to_grayscale(smooth_grad)
        # Save grayscale gradients
        save_gradient_images(grayscale_smooth_grad, file_name_to_export + '_SmoothGrad_gray')
        print('Smooth grad completed')
    except Exception as e:
        print("Couldn't execute: smooth_grad "+str(e))
