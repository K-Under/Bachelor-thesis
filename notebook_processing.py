# -*- coding: utf-8 -*-
"""
Created on Thu 08.02.2024

@author: Leonardo
"""

import numpy as np
from nxs_images import nxs_image



expNumber = "mm34975-1"
edge = "O_K"
path = r"C:\Users\Kevin\Documents\Uni\Physik FS III\Bachelorarbeit\Data\Preprocess_Data\MM34975-1_kevin"
path_dark = r"C:\Users\Kevin\Documents\Uni\Physik FS III\Bachelorarbeit\Data\Preprocess_Data\MM34975-1_kevin"
save_dir = "C:\Users\Kevin\Documents\Uni\Physik FS III\Bachelorarbeit\Data\Preprocessed_data"

runNB = 330720
runDark = 330707

pixel_elastic = 720
slope = 0.06459
#alignRange = [880,1000]

image = nxs_image(runNB = runNB, directory=path, slope=slope, dark_img_run=runDark,directory_dark=path_dark, path_save = save_dir)
image.process_images(pixel_row_start=pixel_elastic-200, pixel_row_stop=pixel_elastic+100)

