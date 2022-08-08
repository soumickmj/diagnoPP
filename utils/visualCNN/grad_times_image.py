"""
Created on Wed Jun 19 17:12:04 2019

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from utils.visualCNN.misc_functions import (get_example_params,
                            convert_to_grayscale,
                            save_gradient_images)
from utils.visualCNN.vanilla_backprop import VanillaBackprop
# from guided_backprop import GuidedBackprop  # To use with guided backprop
# from integrated_gradients import IntegratedGradients  # To use with integrated grads

def execute(model, inputs, classID, file_name_to_export):
    try:
        # Vanilla backprop
        VBP = VanillaBackprop(model)
        # Generate gradients
        vanilla_grads = VBP.generate_gradients(inputs, classID)

        grad_times_image = vanilla_grads[0] * inputs.detach().numpy()[0]
        # Convert to grayscale
        grayscale_vanilla_grads = convert_to_grayscale(grad_times_image)
        # Save grayscale gradients
        save_gradient_images(grayscale_vanilla_grads,
                            file_name_to_export + '_Vanilla_grad_times_image_gray')
        print('Grad times image completed.')
    except Exception as e:
        print("Couldn't execute: Grad times image "+str(e))
