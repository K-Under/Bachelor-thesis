"""
Created on Wed 28.08.2024

@author: Kevin

This file combines the images from evaluation into one image.
It is only tested using the plot_results_script.py file
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

# ======================================
# ===== USER CODE (SET PARAMETERS) =====
# ======================================

#Path to main folder and save folder and the name of the file
root_dir = r"/data/kunter"
save_dir = r"/home/kunter/projects/rixs_ml/evaluating_plotting/imgs/folder"  
filename = [r"/p1",r"/p2",r"/p4",r"/p8"] #list of names for the images or folders 

#List of models we want to plot
model_1 = r"/irunet_images128x128_sum1_kernel_3x3_strides_2x2_apha0p2"
model_2 = r"/irunet_images128x128_sum2_kernel_3x3_strides_2x2_apha0p2"
model_3 = r"/irunet_images128x128_sum4_kernel_3x3_strides_2x2_apha0p2"
model_4 = r"/irunet_images128x128_sum8_kernel_3x3_strides_2x2_apha0p2"

#Do the models have submodels (i.e. in one training session, multiple models have been trained)? If yes, only need to enter the main model below, otherwise enter EACH (sub)model below
submod_bool = True

#if the images should have a different order, one can change this list
order = [1,5,3,4,2]

# ===================================
# ===== MAIN CODE (DO NOT EDIT) =====
# ===================================


#compiling the strings into a list
model_list = [globals()[f'model_{i}'] for i in range(1, 100) if globals().get(f'model_{i}') is not None]


if submod_bool == False:
    """
    This part of the code is executed if the models do not have "submodels".
    Params used:
    - root_dir (str): Path to the main folder
    - model_list (list): List of the models
    """



    #going through each model
    for mod_int,model in enumerate(model_list): #mod_int keeps track of which model we are currently at

        img_dir = root_dir + model + r"/images"  #path to image folder

        if os.path.isdir(img_dir): #checking if image folder exists
            img_dir = img_dir +r"/"

            #we add all of the image paths to a list and then plot the images together
            img_list = os.listdir(img_dir)
            
            #adding the image paths and images
            img_paths = [img_dir + img for img in img_list if os.path.isfile(img_dir + img)] #we check if the path is a real file
            images = [plt.imread(img) for img in img_paths] #getting the image from the paths

            #creating the canvas
            fig = plt.figure(figsize=(19.20,10.80))
            rows = 2
            cols = 3

            #plotting
            for i,img in enumerate(images, 1):
                print("printing image: ", i)
                fig.add_subplot(rows, cols, i)
                plt.imshow(img)
            
            #save image and check if directory exists
            if not os.path.exists(save_dir + filename[mod_int]):
                print("Creating folder...")
                os.makedirs(save_dir + filename[mod_int])
            plt.savefig(save_dir + filename[mod_int] + ".png")
            print(f"Saved in: {save_dir + filename[mod_int]}")

                
else:
    """
    This part of the code is executed if the models have "submodels".
    Params used:
    - root_dir (str): Path to the main folder
    - model_list (list): List of the models
    - dirlist (list): List of the submodels of a model
    """

    #going through each model
    for mod_int,model in enumerate(model_list): #mod_int keeps track of which model we are currently at

        # going through the submodels of the model
        dirlist = os.listdir(root_dir+model)

        for dir in dirlist:
            dir = r"/" + dir #this is necessary because the os.listdir function does not return the \ in the beginning of the string
            img_dir = root_dir + model + dir + r"/images"  #path to image folder

            if os.path.isdir(img_dir): #checking if image folder exists
                img_dir = img_dir +r"/"

                
                #list with the image paths
                img_list = os.listdir(img_dir)
                
                #we add all of the image paths to a list and then plot the images together
                #adding the image paths and images
                img_paths = [img_dir + img for img in img_list if os.path.isfile(img_dir + img)] #we check if the path is a real file
                images = [plt.imread(img) for img in img_paths] #getting the image from the paths

                #creating the canvas
                fig = plt.figure(figsize=(19.20,10.80))
                rows = 2
                cols = 3

                #plotting
                for i,img in enumerate(images, 1):
                    print("printing image: ", i, dir)
                    fig.add_subplot(rows, cols, order[i-1])
                    plt.axis("off")
                    plt.imshow(img)

                #save image and check if directory exists
                if not os.path.exists(save_dir + filename[mod_int]):
                    print("Creating folder...")
                    os.makedirs(save_dir + filename[mod_int])
                plt.savefig(save_dir + filename[mod_int] + dir + ".png", bbox_inches='tight')
                print(f"Saved in: {save_dir + filename[mod_int] + dir}")

                
