"""
Created on Fri 05.07.2024

@author: Kevin

This file extracts the performance of a model/models and plots it.

There is the option to compare multiple models with each other, as well as their submodels, if they have any.
The comparison is done in a way, that the main model will be compared to another mainmodel, with both having the same submodels.
For example:
    main model property: pool size
    submodels: filter size
    comparison (on x axis) will be the pool size, with both having the same filter size (i.e. model 1: pool size 1, filter size 8, model 2: pool size 2, filter size 8)
    then this is repeated for all filter sizes (and plot them in the same plot)
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt # type: ignore

# ======================================
# ===== USER CODE (SET PARAMETERS) =====
# ======================================

#Path to main folder and save folder and the name of the file
root_dir = r"/data/kunter"
save_dir = r"/home/kunter/projects/rixs_ml/evaluating_plotting/imgs" #if directory does not exist, it will be created
filename = r"/Alpha_2Dmae.png"

#Path to the models that should be compared within the main folder. CAN ADD ARBITRARY AMOUNT OF MODELS

model_1 = r"/irunet_images128x128_sum8_ep80_b2_kernel_3x3_strides_2x2_apha0p0/filter_8_ep80"
model_2 = r"/irunet_images128x128_sum8_ep80_kernel_3x3_strides_2x2_apha0p2/filter_8_ep80"
model_3 = r"/irunet_images128x128_sum8_ep80_b2_kernel_3x3_strides_2x2_apha0p4/filter_8_ep80"
model_4 = r"/irunet_images128x128_sum8_ep80_b2_kernel_3x3_strides_2x2_apha0p6/filter_8_ep80"
model_5 = r"/irunet_images128x128_sum8_ep80_b2_kernel_3x3_strides_2x2_apha0p8/filter_8_ep80"
model_6 = r"/irunet_images128x128_sum8_ep80_b2_kernel_3x3_strides_2x2_apha1p0/filter_8_ep80"


#x labels,title and ticks, i.e. the property that is compared. Make sure that the amount of ticks is the same as the amount of models
x_label = "Alpha value"
x_ticks = ["0.0","0.2","0.4","0.6","0.8","1.0"]
title = "Comparison of 2D mae, alpha value"

#Do the models have submodels (i.e. in one training session, multiple models have been trained)? If yes, only need to enter the main model above, otherwise enter EACH (sub)model above
set_bool = False

#submodel ammount (if set_bool = True). ALl of the models need to have the same amount of submodels
subnr = 4

#incase the model has a reduction via pooling, enter the pool size/s here. The list entries need to be equal to the amount of models.
#if not needed comment out
pool_size = [8,8,8,8,8,8]
pool_size_int = 8 #can be used incase all of the models have the same pool size, this can stay uncommented.

# ===================================
# ===== MAIN CODE (DO NOT EDIT) =====
# ===================================

#compiling the strings into a list
model_list = [globals()[f'model_{i}'] for i in range(1, 100) if globals().get(f'model_{i}') is not None]

#plot function
def plot_mae(Fmean_list, Flabel_list = None):
    """
    This function plots the mae of each model
    Params:
    -Fmean_list: List with mean of each mae
    -Flabel_list, list with labels
    """
    #calling global variables
    global x_label
    global x_ticks
    global save_dir
    global filename
    global set_bool
    global title

    #preparing x-axis
    x_axis = np.arange(0, len(x_ticks))
    plt.xticks(x_axis, x_ticks)

    #checking if there are submodels, if yes plot each one, if not, plot only the meanlist
    if set_bool == True:
        for i in range(len(Fmean_list)):
            plt.plot(x_axis, Fmean_list[i], label=Flabel_list[i], linewidth=0.5, marker='o')
    else:
        plt.plot(x_axis, Fmean_list, linewidth=0.5, marker='o')

    #finishing the plot
    plt.xlabel(x_label)
    plt.ylabel("Mean Absolute Percentage Error")
    plt.grid()
    plt.legend()
    plt.title(title)
    plt.savefig(save_dir + filename, bbox_inches='tight')
    print("saved as: ", save_dir, filename, "\n\n")
    return None


if set_bool == False:
    """
    This part of the code is executed if the models do not have "submodels".
    Params used:
    - root_dir (str): Path to the main folder
    - model_list (list): List of the models that will be compared with each other
    - mean_list (list): List of the mean values of the mae of the models
    """

    mean_list = []
    #extracting the hdf5 datafiles by going through each model
    for mod_int,model in enumerate(model_list):
        #adding the mean of the mae to the list
        with h5py.File(str(root_dir+model+r"/evaluation.hdf5"), 'r') as hdf_file:

            print("evaluating model: ", model)
            if 'pool_size' in locals():
                mean_list.append(np.mean(hdf_file["mae"][:])/pool_size[mod_int]) # type: ignore
            else:
                mean_list.append(np.mean(hdf_file["mae"][:])/pool_size_int)

    #plotting the mae of the model
    plot_mae(mean_list)

else:
    """
    This part of the code is executed if the models have "submodels".
    Params used:
    - root_dir (str): Path to the main folder
    - model_list (list): List of the models that will be compared with each other, these now have "submodels"
    - mean_list (list): List of the mean values of the mae of the models, these now have sublists for the submodels
    - dirlist (list): List of the submodels of a model
    """
    #The mean list, with sublists for the submodels.
    mean_list = [[] for n in range(subnr)]

    #going through each model
    for mod_int,model in enumerate(model_list):

        # going through the submodels of the model
        dirlist = os.listdir(root_dir+model)
        label_list = []

        i = 0
        for dir in dirlist:
            dir = r"/" + dir #this is necessary because the os.listdir function does not return the \ in the beginning of the string
            # adding the mean of the mae to the list
            if os.path.isfile(root_dir+model+dir+r"/evaluation.hdf5"):

                with h5py.File(str(root_dir+model+dir+r"/evaluation.hdf5")) as hdf_file:
                    print("evaluating model: ", model + dir)
                    if 'pool_size' in locals():
                        mean_list[i].append(np.mean(hdf_file["mae"][:])/pool_size[mod_int]) # type: ignore
                    else:
                        mean_list[i].append(np.mean(hdf_file["mae"][:])/pool_size_int)
                    label_list.append(dir)
                    i = i+1 #the counter only goes up if we actually found and used a submodel      

    #plotting the mae of the models
    plot_mae(mean_list, label_list)
