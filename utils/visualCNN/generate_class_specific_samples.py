"""
Created on Thu Oct 26 14:19:44 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np

import torch
from torch.optim import SGD
from torchvision import models

try:
    from utils.visualCNN.misc_functions import preprocess_image, recreate_image, save_image
except:
    from misc_functions import preprocess_image, recreate_image, save_image


class ClassSpecificImageGeneration():
    """
        Produces an image that maximizes a certain class with gradient ascent
    """
    def __init__(self, model, target_class, output_path=None, in_shape=(224, 224, 3), mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225]):
        self.mean = mean
        self.std = std
        self.model = model
        self.model.eval()
        self.target_class = target_class
        # Generate a random image
        self.created_image = np.uint8(np.random.uniform(0, 255, in_shape))
        # Create the folder to export images if not exists
        if output_path is None:
            self.output_path = '../generated/class_'+str(self.target_class)
        else:
            self.output_path = os.path.join(output_path, 'ClassSpecificImageGeneration', 'class_'+str(self.target_class))
        os.makedirs(self.output_path, exist_ok=True)

    def generate(self, iterations=150):
        """Generates class specific image

        Keyword Arguments:
            iterations {int} -- Total iterations for gradient ascent (default: {150})

        Returns:
            np.ndarray -- Final maximally activated class image
        """
        initial_learning_rate = 6
        for i in range(1, iterations):
            # Process image and return variable
            self.processed_image = preprocess_image(self.created_image, False)

            #send processed_image to the correct device
            self.processed_image = self.processed_image.to(next(self.model.parameters()).device).detach().requires_grad_(True)
            #we want this to be a leaf Teansor. But, toDevice creates a copy and makes it a non-leaf Tensor. So, detach and then requires grad has to be done

            # Define optimizer for the image
            optimizer = SGD([self.processed_image], lr=initial_learning_rate)
            # Forward
            output = self.model(self.processed_image)
            # Target specific class
            class_loss = -output[0, self.target_class]

            if i % 10 == 0 or i == iterations-1:
                print('Iteration:', str(i), 'Loss',
                      "{0:.2f}".format(class_loss.cpu().data.numpy()))
            # Zero grads
            self.model.zero_grad()
            # Backward
            class_loss.backward()
            # Update image
            optimizer.step()
            # Recreate image
            self.created_image = recreate_image(self.processed_image)
            if i % 10 == 0 or i == iterations-1:
                # Save image
                im_path = os.path.join(self.output_path, 'c_'+str(self.target_class)+'_'+'iter_'+str(i)+'.png')
                save_image(self.created_image, im_path)

        return self.processed_image


def execute(model, classID, outpath=None, iterations=150, in_shape=(512,512,3)):
    try:
        csig = ClassSpecificImageGeneration(model, classID, outpath, in_shape)
        csig.generate(iterations=iterations)
    except Exception as e:
        print("Couldn't execute: ClassSpecificImageGeneration"+str(e))
    
if __name__ == "__main__" :
    target_class = 130  # Flamingo
    pretrained_model = models.alexnet(pretrained=True)
    pretrained_model.cuda()
    execute(pretrained_model, target_class, None, 150)
    # csig = ClassSpecificImageGeneration(pretrained_model, target_class)
    # csig.generate(iterations=150)
