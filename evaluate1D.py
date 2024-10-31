"""
Created on Sat 31.08.2024

@author: Kevin

This file extracts the performance of a model from the 1D predicted spectrum and plots it.
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt # type: ignore

import sys
PATH = "/home/kunter/repos/lqmr_deep_learning/src" 
sys.path.append(PATH)
import rixs
import misc_rixs as mr



# ======================================
# ===== USER CODE (SET PARAMETERS) =====
# ======================================

#Path to main folder and save folder and the name of the file
root_dir = r"/data/kunter"
save_dir = r"/home/kunter/projects/rixs_ml/evaluating_plotting/imgs/"
filename = r"/Alpha_1Dmaezoom"

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
title = "Alpha comparison 1D elastic peak"

#incase the model has a reduction via pooling, enter the pool size/s here. The list entries need to be equal to the amount of models.
#if not needed comment out
#pool_size = [1,2,4,8]
pool_size_int = 8 #can be used incase all of the models have the same pool size, this can stay uncommented.

#Do the models have submodels (i.e. in one training session, multiple models have been trained)? If yes, only need to enter the main model below, otherwise enter EACH (sub)model below
set_bool = False

#submodel ammount (if set_bool = True)
subnr = 4

####################################
#The below parameters are optional.# 
####################################

#print spectra as well
spec_plot_bool = False
zoom_bool = True #print spectra zoomed in on the elastic peak. Also plots the mae of the last 128 pixels from the elastic peak (works also with spec_plot_bool = False)
spec_save_name = "_spectra"

#print multiple spectra in one plot
multi_spec_bool = False

# ===================================
# ===== MAIN CODE (DO NOT EDIT) =====
# ===================================

#function part
def mae1D (predict, actual, Fpool_size = 1): #function for calculating the mae of a 1D spectra
    mean_list = []
    for spectrum in range(len(actual[0])):
        mae = 0
        for i in range(len(actual)):
            mae += abs(actual[i][spectrum]-predict[i][spectrum])/Fpool_size
        mean_list.append(mae/len(actual))
    return mean_list

def mae1D_zoom(predict, actual, Fpool_size = 1): #function for calculating the mae of a 1D spectra
    mean_list = []
    for spectrum in range(len(actual)):
        mae = 0
        for i in range(len(actual[0])):
            mae += abs(actual[spectrum][i]-predict[spectrum][i])/Fpool_size
        mean_list.append(mae/len(actual[0]))
    return mean_list

def plot_mae1D(Fmean_list, Flabel_list = None):
    """
    This function plots the 1D mae
    params:
    - Fmean_list (list): list of the mean values of the mae of the models
    - Flabel_list (list): list of the labels of the models
    """
    #get the global variables
    global x_ticks
    global x_label
    global save_dir
    global filename
    global set_bool
    global title

    print("\nPlotting the mae: \n")
    x_axis = np.arange(0, len(x_ticks))
    plt.xticks(x_axis, x_ticks)
    # check if model has submodels
    if set_bool == True:
        for i in range(len(Fmean_list)):
            print("plotting: ", label_list[i])
            plt.plot(x_axis, Fmean_list[i], label=Flabel_list[i], linewidth=0.5, marker='o')
    else:
        plt.plot(x_axis, Fmean_list, linewidth=0.5, marker='o')

    #finishing the plot
    plt.xlabel(x_label)
    plt.ylabel("Mean Absolute Percentage Error")
    plt.grid()
    plt.legend()
    plt.title(title)
    #plt.savefig(save_dir + filename + r"/" + filename + ".png", bbox_inches='tight')
    plt.savefig(save_dir + filename + ".png", bbox_inches='tight')
    print("saved as: ", save_dir, filename, "\n\n")
    return None

def zoom_data(Fhdf_file, Tpool_size = 1):
    """
    This function zooms in on the elastic peak of the spectra
    params:
    - Fhdf_file (h5py file): hdf5 file
    """
    print("zooming...")
    #getting the metadata to acquire the run number of each spectra
    meta_group = Fhdf_file.get('/meta')
    sub_item = meta_group["runs"]
    runs = sub_item[()]
    #empty list of lists for zoomed in data
    elastic_data_hc = [[] for n in range(len(Fhdf_file["dn"]["data"][0]))]
    elastic_data_dn = [[] for n in range(len(Fhdf_file["dn"]["data"][0]))]
    for j in range(len(Fhdf_file["dn"]["data"][0])):
        #getting the elastic peak using the run number and filling it into the list
        stop = mr.getLimits(runs[j],imgs=Fhdf_file["hc"]["data"][:,j],ySize=2048)[1]-mr.getLimits(runs[j],imgs=Fhdf_file["hc"]["data"][:,j],ySize=2048)[0]
        start = stop - 128
        elastic_data_hc[j] = Fhdf_file["hc"]["data"][start:stop,j]
        elastic_data_dn[j] = Fhdf_file["dn"]["data"][start:stop,j]
    Fmae_list = mae1D_zoom(elastic_data_dn, elastic_data_hc, Tpool_size) #calculating the mae
    return Fmae_list #returning the mae list

def plot_spec(Fhc, Flc, Fdn, Fxaxis, FrunNB, FimgNB, Fplotint, Fdir = ""):
    """
    This function plots the spectra
    params:
    - Fhc (np.array): high count data
    - Flc (np.array): low count data
    - Fdn (np.array): denoised data
    - Fxaxis (np.array): x axis of the spectra
    - FrunNB (int): run number
    - FimgNB (int): image number
    - Fplotint (int): plot number
    - dir (str): submodel directory
    """
    global zoom_bool
    #plotting hc,lc and dn data for different spectra
    start, stop = 0, None

    if zoom_bool == True: #if we want to zoom in, we change the start and stop varaibles
        stop = mr.getLimits(runNB=FrunNB,imgs=Fhc[:,FimgNB],ySize=2048)[1]-mr.getLimits(runNB=FrunNB,imgs=Fhc[:,FimgNB],ySize=2048)[0]
        start = stop-128
        x_ticks_spec = range(start,stop)
        plt.xticks(Fxaxis, x_ticks_spec)
        plt.locator_params(nbins=4)
    print("printing image: ", Fplotint, Fdir)
    plt.plot(Fxaxis,Fhc[start:stop,FimgNB], label="hc")
    plt.plot(Fxaxis,Flc[start:stop,FimgNB], label="lc")
    plt.plot(Fxaxis,Fdn[start:stop,FimgNB], label="dn")
    plt.title(f"img {FimgNB}, runNB {FrunNB}")
    plt.grid()
    plt.legend()
    return None

def init_plot_spec(Fhdf_file, fdir = ""):

    global zoom_bool

    hc_data = Fhdf_file["hc"]["data"][:]
    lc_data = Fhdf_file["lc"]["data"][:]
    dn_data = Fhdf_file["dn"]["data"][:]

    fig = plt.figure(figsize=(19.20,10.80))
    rows = 3
    cols = 1
    x_axis_spec = np.arange(0, len(hc_data[:,0]))
    #start, stop = 0, None

    if zoom_bool == True: #if we want to zoom in, we use this
        x_axis_spec = np.arange(0, 128)

    #plotting hc,lc and dn data for different spectra
    Fax1 = fig.add_subplot(rows, cols, 1)
    plot_spec(Fhc=hc_data,Flc=lc_data,Fdn=dn_data,Fxaxis=x_axis_spec,FrunNB=184999,FimgNB=33,Fplotint=1,Fdir=fdir)

    Fax2 = fig.add_subplot(rows, cols, 2)
    plot_spec(Fhc=hc_data,Flc=lc_data,Fdn=dn_data,Fxaxis=x_axis_spec,FrunNB=184985,FimgNB=13,Fplotint=2,Fdir=fdir)

    Fax3 = fig.add_subplot(rows, cols, 3)
    plot_spec(Fhc=hc_data,Flc=lc_data,Fdn=dn_data,Fxaxis=x_axis_spec,FrunNB=330921,FimgNB=321,Fplotint=3,Fdir=fdir)

    return Fax1, Fax2, Fax3

def plot_multi_spec(Fhc, Fdn, Fax, Fxaxis, FrunNB, FimgNB, Fplotint, Flabel = "", Fdir = ""):
    """
    This adds additional spectra onto an existing one
    params:
    - Fhc (np.array): high count data
    - Fdn (np.array): denoised data
    - Fax (plt.axis): axis of the plot
    - Fxaxis (np.array): x axis of the spectra
    - FrunNB (int): run number
    - FimgNB (int): image number
    - Fplotint (int): plot number
    - Flabel (str): label of the plot
    - dir (str): submodel directory
    """

    global zoom_bool
    #plotting hc,lc and dn data for different spectra
    start, stop = 0, None

    if zoom_bool == True: #if we want to zoom in, we change the start and stop varaibles
        stop = mr.getLimits(runNB=FrunNB,imgs=Fhc[:,FimgNB],ySize=2048)[1]-mr.getLimits(runNB=FrunNB,imgs=Fhc[:,FimgNB],ySize=2048)[0]
        start = stop-128
    print("     Adding spectra to: ", Fplotint, Fdir)
    Fax.plot(Fxaxis,Fdn[start:stop,FimgNB], label = Flabel)

    return None


#compiling the strings into a list
model_list = [globals()[f'model_{i}'] for i in range(1, 100) if globals().get(f'model_{i}') is not None]
#code execuction

if not os.path.exists(save_dir): #creating a directory, if one does not yet exist
    os.makedirs(save_dir)

if set_bool == False:
    """
    This part of the code is executed if the models do not have "submodels".
    Params used:
    - root_dir (str): Path to the main folder
    - model_list (list): List of the models that will be compared with each other
    - mean_list (list): List of the mean values of the mae of the models
    """

    mean_list = []

    #going through each model
    for mod_int, model in enumerate(model_list):

        filelist = os.listdir(root_dir + model)
        for file in filelist: #going through the files within the model
            file = r"/" + file
            if "spectra1D.hdf5" not in file: #checking if the file is the 1D spectra. if not, continue with the next file
                continue
            with h5py.File(str(root_dir + model + file)) as hdf_file: #opening the file
                print("evaluating model: ", model)

                #calculating the mae w.r.t zoom or no zoom
                if zoom_bool == True: #if only interested in the elastic peak, we need to get these individual peaks
                    mae_list = zoom_data(hdf_file, pool_size[mod_int] if 'pool_size' in locals() else pool_size_int)
                else:
                    mae_list = mae1D(hdf_file["dn"]["data"][:],hdf_file["hc"]["data"][:], pool_size[mod_int] if 'pool_size' in locals() else pool_size_int) #list of the 1D mae between denoised and high count
                mean_list.append(np.mean(mae_list)) #adding the mean of the maes to the mean list
                #plotting the spectra
                if spec_plot_bool == True: 
                    print("\nPrinting spectra\n")
                    init_plot_spec(hdf_file)
                    
                    #saving the plots
                    plt.suptitle(f"Model: {model}", fontsize=14)
                    plt.savefig(save_dir + r"/model_" + str(mod_int) + spec_save_name + ".png")
                    print("spectra saved as: ", save_dir, r"/model_", spec_save_name + "\n")
                    plt.close()
                
                #plotting spectra overlay
                if multi_spec_bool == True:
                    print("\nprinting multispectra\n")
                    if mod_int == 0:
                        print("Preparing multiplot...")
                        ax1, ax2, ax3 = init_plot_spec(hdf_file)
                        
                    else:
                        hc_data = hdf_file["hc"]["data"][:]
                        dn_data = hdf_file["dn"]["data"][:]

                        x_axis_spec = np.arange(0, len(hc_data[:,0]))
                        if zoom_bool == True: #if we want to zoom in, we use this
                            x_axis_spec = np.arange(0, 128)

                        plot_multi_spec(hc_data,dn_data,ax1,x_axis_spec,184999,33,1,"model: "+str(model))
                        plot_multi_spec(hc_data,dn_data,ax2,x_axis_spec,184985,13,2,"model: "+str(model))
                        plot_multi_spec(hc_data,dn_data,ax3,x_axis_spec,330921,321,3,"model: "+str(model))
                        

                        #save the plot, if its the last part
                    if mod_int == len(model_list)-1:
                        ax1.legend()
                        ax2.legend()
                        ax3.legend()
                        plt.suptitle(f"Spectra overlay of the different models", fontsize=14)
                        plt.savefig(save_dir + r"/multispectra" + spec_save_name + ".png")
                        print("multispectra saved as: ", save_dir, r"/multispectra", spec_save_name + "\n")
                        plt.close()

                        
    #plotting the mae of the model
    plot_mae1D(mean_list)
    print("FINISHED")

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
    for mod_int, model in enumerate(model_list):

        # going through the submodels of the model
        dirlist = os.listdir(root_dir+model)
        label_list = []

        i = 0 #integer for keeping track of the submodel

        for dir in dirlist: #dir represents a submodel in this case, but can also be other directories if they are present
            dir = r"/" + dir #this is necessary because the os.listdir function does not return the \ in the beginning of the string

            if os.path.isdir(root_dir + model + dir) == False: #if path is not a dir (submodel), skip to next file
                continue

            filelist = os.listdir(root_dir + model + dir)
            for file in filelist: #going through the files within the submodel
                file = r"/" + file
                if "spectra1D.hdf5" not in file: #checking if the file is the 1D spectra. if not, continue with the next file
                    continue
                with h5py.File(str(root_dir + model + dir + file)) as hdf_file: #opening the file
                    print("evaluating model: ", model + dir)

                    #calculating the mae w.r.t zoom or no zoom
                    if zoom_bool == True: #if only interested in the elastic peak, we need to get these individual peaks
                        mae_list = zoom_data(hdf_file, pool_size[mod_int] if 'pool_size' in locals() else pool_size_int)
                    else:
                        mae_list = mae1D(hdf_file["dn"]["data"][:],hdf_file["hc"]["data"][:], pool_size[mod_int] if 'pool_size' in locals() else pool_size_int) #list of the 1D mae between denoised and high count
                    mean_list[i].append(np.mean(mae_list)) #adding the mean of the maes to the mean list

                    label_list.append(dir)
                    i = i+1 #going to the next submodel
        
                    if spec_plot_bool == True: #plotting a few spectra 
                        print("\nPrinting spectra\n")
                        init_plot_spec(hdf_file,dir)
                           
                        #saving the plots
                        if not os.path.exists(save_dir + r"/model_" + str(mod_int)): #creating a directory, if one does not yet exist
                            os.makedirs(save_dir + r"/model_" + str(mod_int))
                        plt.suptitle(f"Model: {model},  {dir}", fontsize=14)
                        plt.savefig(save_dir + r"/model_" + str(mod_int) + dir + spec_save_name + ".png")
                        print("saved as: ", save_dir, dir, spec_save_name + "\n")
                        plt.close()

    #plotting the mae of the models
    plot_mae1D(mean_list, label_list)
    print("FINISHED")
 
