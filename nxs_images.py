# -*- coding: utf-8 -*-
"""
Created on Thu 08.02.2024

@author: Leonardo
"""

#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import h5py
import pathlib
from scipy.signal import medfilt2d, correlate
from scipy.interpolate import RegularGridInterpolator as rgi
from scipy.ndimage import gaussian_filter
from nexusformat.nexus import *
import os, fnmatch

class nxs_image:
    """
    Basic class for Diamond RIXS images. 
    """
    def __init__(self, runNB, directory, directory_dark = None,
                 imgs=None, slope=None, shifts=None, 
                 dark_img_run=None, dark_poly=None, dark_img = None, dark_img_filtered=None, 
                 imgs_pure = None, imgs_processed = None, path_save = None):
        """
        Class constructor.

        Parameters
        runNB : int, run number of experiment
        directory : str, directory where the data is stored
        directory_dark : str, directory where the dark image is stored
        imgs : array, images
        slope : float, slope of the curvature
        shifts : array, shifts of the images
        dark_img_run : int, run number of the dark image
        dark_img : array, dark image
        dark_img_filtered : array, dark image filtered
        dark_poly : array, dark image parameters
        imgs_pure : array, pure images
        imgs_processed : array, processed images
        path_save : str, path to save the finished images
        """

        self.runNB = str(runNB)
        self.directory = directory
        self.directory_dark = directory_dark if directory_dark is not None else None
        self.imgs = np.array(imgs) if imgs is not None else None
        self.slope = np.array(slope) if slope is not None else None
        self.shifts = np.array(shifts) if shifts is not None else None
        self.dark_img_run = np.array(dark_img_run) if dark_img_run is not None else None
        self.dark_img = np.arry(dark_img) if dark_img is not None else None
        self.dark_img_filtered = np.arry(dark_img_filtered) if dark_img_filtered is not None else None
        self.dark_poly = np.array(dark_poly) if dark_poly is not None else None
        self.imgs_pure = np.array(imgs_pure) if imgs_pure is not None else None
        self.imgs_processed = np.array(imgs_processed) if imgs_processed is not None else None
        self.path_save = path_save if path_save is not None else None

    def process_images(self, pixel_row_start, pixel_row_stop, kernel_size=[5,7], spikes_threshold=0.7,
                       kernel_size_dark = [5,5], sigma_dark = [10,10]):
        """
        this function uses other functions to process the image
        variable info:
        imgs_pure: copy of the images which are going to be overwritten and worked with
        imgs_processed:
        """
        self.get_images_and_parameters() #
        self.get_dark_img() #
        self.remove_spikes_dark(kernel_size = kernel_size_dark, sigma = sigma_dark) #
        self.remove_spikes_img(kernel_size=kernel_size, spikes_threshold=spikes_threshold) #
        self.subtract_dark()#
        self.correct_curvature() #
        self.correct_shift(pixel_row_start, pixel_row_stop)
        self.save_images()

    
    def get_spectrum(self, normalized=True):
        filename = self.find(self.runNB, self.directory)
        f = nxload(filename,mode='r')

        if normalized:
            try:
                energy_loss = f.processed.summary['1-Combined RIXS image reduction']['normalized_correlated_spectrum_0']['Energy loss'].nxvalue
                spectrum = f.processed.summary['1-Combined RIXS image reduction']['normalized_correlated_spectrum_0'].data.nxvalue
            except:
                energy_loss = f.processed.summary['1-RIXS image reduction']['normalized_correlated_spectrum_0']['Energy loss'].nxvalue
                spectrum = f.processed.summary['1-RIXS image reduction']['normalized_correlated_spectrum_0'].data.nxvalue
        else:
            try:
                energy_loss = f.processed.summary['1-Combined RIXS image reduction']['correlated_spectrum_0']['Energy loss'].nxvalue
                spectrum = f.processed.summary['1-Combined RIXS image reduction']['correlated_spectrum_0'].data.nxvalue       
            except:
                energy_loss = f.processed.summary['1-RIXS image reduction']['correlated_spectrum_0']['Energy loss'].nxvalue
                spectrum = f.processed.summary['1-RIXS image reduction']['correlated_spectrum_0'].data.nxvalue
                    
        return energy_loss, spectrum
    
    def get_images_and_parameters(self): #loading image and getting the relevant parameters
        """
        This function loads the image and relevant parameters such as alpha,beta, shifts and counting time
        """
        filename = self.find(self.runNB,self.directory) #getting the file with the run number
        f = nxload(filename,mode='r') #f is the file

        try:
            self.imgs = f.entry['andor']['data'].nxvalue #the file itself
        except:
            self.imgs = f.entry1['andor']['data'].nxvalue

        self.imgs_processed = self.imgs.copy() #Copy for correlation later
        self.imgs_pure = self.imgs.copy() #copy which will be overwritten and used later
        dark_offset = f.processed.auxiliary['0-Image background subtraction - Fitted to a PDF']['dark_offset_0'].data.nxvalue
        dark_offset = np.array(dark_offset) #Beta
        dark_scale = f.processed.auxiliary['0-Image background subtraction - Fitted to a PDF']['dark_scale_0'].data.nxvalue
        dark_scale = np.array(dark_scale) #Alpha
        self.dark_poly= np.array([dark_scale, dark_offset])
        try:
            self.shifts = f.processed.summary['1-RIXS image reduction']['correlated_shift_0'].data.nxvalue #shift value
        except:
            self.shifts = f.processed.summary['1-Combined RIXS image reduction']['correlated_shift_0'].data.nxvalue
            
        try:
            self.count_time = f.entry['instrument']['m4c1']['count_time'].nxvalue #count time
        except:
            try:
                self.count_time = f.entry1['instrument']['m4c1']['count_time'].nxvalue
            except:
                self.count_time = 500

    def get_dark_img(self): #backround
        """
        This function loads the dark image and its parameters
        """
        filename = self.find(self.dark_img_run,self.directory_dark)
        f = nxload(filename,mode='r') #loading background
        try:
            self.dark_img = f.entry['andor']['data'].nxvalue #the file itself?
        except:
            self.dark_img = f.entry1['andor']['data'].nxvalue   

        self.dark_img = self.dark_img[0,:,:].astype(float) #?
        try:
            self.count_time = f.entry['instrument']['m4c1']['count_time'].nxvalue #count time
        except:
            try:
                self.count_time = f.entry1['instrument']['m4c1']['count_time'].nxvalue
            except:
                self.count_time = 500

    def subtract_dark(self):
        """
        this function subtracts the dark image from the images
        return: imgs_pure using imgs_pure = imgs - alpha*dark_img - beta
        """

        self.imgs_pure = self.imgs_pure.astype(float) #converting the images from unit16 to float
        for num_image in range(0,self.imgs_pure.shape[0]):
            self.imgs_pure[num_image] = self.imgs_pure[num_image] - (self.dark_poly[0,num_image]*self.dark_img_filtered + self.dark_poly[1,num_image])

    def correct_curvature(self):
        """
        this function corrects the curvature of the images
        """

        if self.slope is None:
            raise Exception("No curvature defined")
        
        for num_image in range(0,self.imgs.shape[0]):

            xdim,ydim=self.imgs_pure[num_image,:,:].shape
            x=np.arange(xdim+1)
            y=np.arange(ydim+1)
            xx,yy=np.meshgrid(x[:-1]+0.5,y[:-1]+0.5)
            #xxn=xx-curv[0]*yy-curv[1]*yy**2
            xxn=xx-self.slope*yy #correcting the curvature
            #yyn=yy-curv*xx

            self.imgs_pure[num_image,:,:] = np.histogramdd((xxn.flatten(),yy.flatten()),bins=[y,x],weights=self.imgs_pure[num_image,:,:].T.flatten())[0]

            #im_corr = np.histogramdd((xx.flatten(),yyn.flatten()),bins=[y,x],weights=im.T.flatten())[0]#.T
            #print(ret.shape)

    def correct_shift(self, pixel_row_start, pixel_row_stop):
        """
        this function corrects the shift of the images

        params:
        pixel_row_start: int, the starting pixel
        pixel_row_stop: int, the stopping pixel

        """
        self.shifts = np.zeros((self.imgs.shape[0],1))

        for num_image in range(0,self.imgs.shape[0]):
            self.shifts[num_image] = self.correlate_spectra(self.imgs_pure[num_image,:,:],
                                                           self.imgs_pure[0,:,:], pixel_row_start, pixel_row_stop) #correlating the spectra and getting the lag value
            if abs(self.shifts[num_image])>0.1: #if the shift is too large, we shift the image
                xdim,ydim=self.imgs[num_image,:,:].shape
                x=np.arange(xdim)
                y=np.arange(ydim)
                interp = rgi((x-self.shifts[num_image], y), self.imgs_pure[num_image,:,:],bounds_error=False, fill_value=0)
                
                xx,yy=np.meshgrid(x,y)
                self.imgs_pure[num_image,:,:] = interp((xx,yy)).T #shifting the images
            else:
                self.imgs_pure[num_image,:,:] = self.imgs_pure[num_image,:,:] #if the shift is too small, we don't shift the image

    def remove_spikes_dark(self, kernel_size = [5,5], sigma = [10,10]):
        """
        this function uses a median filter to remove spikes from the dark image

        params:
        kernel_size: list, the size of the kernel 2D
        sigma: list, the sigma for the gaussian filter
        """
        if self.count_time is None:
            raise Exception("Provide counting time first.")
        dark_med = medfilt2d(self.dark_img, kernel_size) #applying the median filter
        spikes_dark = self.dark_img - dark_med #getting the spikes
        spikes_dark = spikes_dark / self.count_time
        dark_filt = np.where(spikes_dark>1.0, dark_med, self.dark_img) #filtering the spikes
        dark_filt2 = gaussian_filter(dark_filt, sigma)
        self.dark_img_filtered = dark_filt2 / (dark_filt2.sum() / self.dark_img.sum()) #normalizing the dark image

    def remove_spikes_img(self, kernel_size=[5,7], spikes_threshold=0.7):
        """
        this function uses a median filter to remove spikes from the images

        params:
        kernel_size: list, the size of the kernel 2D
        spikes_threshold: float, the threshold for the spikes
        """
        if self.count_time is None:
            raise Exception("Provide counting time first.")
        
        for num_image in range(0,self.imgs.shape[0]): #looping through the images
            img_corr_med = medfilt2d(self.imgs_pure[num_image,:,:], kernel_size) #applying the median filter
            spikes = self.imgs_pure[num_image,:,:] - img_corr_med #getting the spikes
            spikes = spikes / self.count_time #normalizing the spikes
            self.imgs_pure[num_image,:,:] = np.where(spikes>spikes_threshold, img_corr_med ,self.imgs_pure[num_image,:,:]) #filtering the spikes

    def plot_image(self):
        """
        this function plots the images
        CURRENTLY NOT WORKING (variable names changed)
        """

        num_image = 0
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
        ax1.imshow(self.imgs_pure[num_image],
                   vmin=self.imgs_pure[num_image].mean()-1*self.imgs_pure[num_image].std(),
                    vmax=self.imgs_pure[num_image].mean()+3*self.imgs_pure[num_image].std(),cmap = 'viridis')
        ax1.set_title('Background subtracted')
        ax1.imshow(self.imgs_processed[num_image],
                   vmin=self.imgs_processed[num_image].mean()-1*self.imgs_processed[num_image].std(),
                    vmax=self.imgs_processed[num_image].mean()+3*self.imgs_processed[num_image].std(),cmap = 'viridis')
        ax2.set_title('Slope corrected, shifted')
        #ax3.imshow(spikes,cmap = 'viridis', vmin=1.0, vmax=1.1)
        ax3.imshow(self.imgs_processed[num_image],cmap = 'viridis', vmin=0, vmax=10)
        ax3.set_title('Spikes removed and shifted')
        ax4.plot(self.imgs_processed[num_image].mean(axis=1))
        ax4.plot(self.imgs_pure[num_image].mean(axis=1))
        ax3.set_title('Final spectrum')
        plt.show()
        
        plt.figure()
        plt.plot(self.dark_img.mean(axis=1), label='dark profile')
        plt.plot(self.dark_img_filtered.mean(axis=1), label='dark filtered')
        plt.plot(self.get_dark_profile()[num_image,:], label='Diamond')
        plt.title('Background subtraction')
        plt.legend()
        plt.show()

    @staticmethod
    def correlate_spectra(spec, spec_ref, pixel_start, pixel_stop):
        """
        this function correlates two spectra and returns the lag value to be used for shifting the images

        params:
        spec: the spectrum to be shifted
        spec_ref: the reference spectrum
        pixel_start: the starting pixel
        pixel_stop: the stopping pixel

        return:
        lag: the lag value to be used for shifting the images
        """
        spec = spec.mean(axis=1) #mean of the spectrum along the Intensity axis
        spec_ref = spec_ref.mean(axis=1) #mean of the reference spectrum along the Intensity axis
        xx = np.arange(pixel_start, pixel_stop+0.25,0.25)
        spec = np.interp(xx, np.arange(0,len(spec)), spec) #interpolating the spectrum
        spec_ref = np.interp(xx, np.arange(0,len(spec_ref)), spec_ref) #interpolating the reference spectrum
        crosscorr = correlate(spec, spec_ref)
        lag_values = np.arange(-spec.shape[0]+1,spec.shape[0], 1)
        lag = lag_values[np.argmax(crosscorr)]/4

        return lag

    def find(self, number, path):
        """
        this function finds a file in the directory where the experimental data is defined to be

        params:
        self: class object,  used for getting the directory = path
        number: int, the run number of the file to be found.
        path: str, the directory where the file is to be found

        return:
        result[-1]: str, the last file found in the directory
        """
        result = []
        number=str(number)
        #path = self.directory
        for root, dirname, files in os.walk(path): #we walk through the files using os.walk, where the folder to be analized is defined with the path (directory = root)
            for name in files:
                if fnmatch.fnmatch(name, '*'+number+'*'):
                    result.append(os.path.join(root, name))

        if len(result)>1.1:
            print("Warning: more than one file has been found. \n   Used file:", result[-1], "\n")
        elif len(result)<0.9:
            print("Warning: no file has been found. \n")

            """
            This code was used to write the run number of the file which was not found to a text file filled with offsets
            
            #f = open(r"C:\Users\Kevin\Documents\Uni\Physik FS III\Bachelorarbeit\Data\vertical_offset.txt", "a") #Path to files
            #f.write("runNB: " + str(self.runNB) + "  NO DATA FOUND" + "\n")
            #f.close()
            """

            raise Exception("No file found")
        return result[-1] #we return the last file in the bunch

    def save_images(self):
        """
        this function saves the images
        """

        #we compare the mean of the spectra of our images and the spectra given by the experiemental data by making a subplot
        plt.plot(self.imgs_pure.mean(axis=0).mean(axis=1), label='mean of all images')
        #making x axis for spectra, shifted by 10 pixels
        x_axis=np.arange(10,self.get_spectrum()[1].shape[0]+10,1)
        plt.plot(x_axis,self.get_spectrum()[1]*5+2.5, label='spectrum from data')
        plt.legend()
        plt.show()
        plt.clf()
        plt.imshow(self.imgs_pure.mean(axis=0), cmap='viridis')
        plt.show()
        plt.clf()
        #comparing all the spectra of the subimages
        for i in range(0,self.imgs_pure.shape[0]):
            plt.plot(self.imgs_pure[i,:,:].mean(axis=1), label='image '+str(i))
        plt.show()

        """
        Code used to write the vertical offset to a text file
        
        #writing vertical offset to a text file
        f = open(r"C:\Users\Kevin\Documents\Uni\Physik FS III\Bachelorarbeit\Data\vertical_offset.txt", "a")
        f.write("runNB: "+str(self.runNB)+"  v-offset: "+str(np.mean(self.imgs_pure.mean(axis=0).mean(axis=1)[1500:1800]))+"\n")
        f.close()
        """

        #saving the images as an hdf5 file, insert your path here
        filename_save = pathlib.Path(self.path_save) / ('runNB_' + str(self.runNB) + "_32.hdf5")
        hf = h5py.File(filename_save, 'w')

        hf.create_dataset('data',data=self.imgs_pure, dtype='float32')
        hf.attrs['run_number'] = str(self.runNB)
        hf.close()
