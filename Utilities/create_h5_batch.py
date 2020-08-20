# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 13:46:50 2020

@author: Di Lorenzo Tech
"""

import os
from pypl2.Autosorting import pl2_to_h5

pl2_dir=r'R:\Dannymarsh Sorting Emporium\Single Channel Resorts\to-be resorted'
min_licks=1000  
overwrite_h5=0 #change to 0 if you dont want to overwrite h5

runfiles=os.listdir(pl2_dir)
for file in runfiles:
    if file.endswith('.pl2'):
        if os.path.isfile(pl2_dir+'/'+file[:-4]+'.h5') and overwrite_h5==1:
            os.remove(pl2_dir+'/'+file[:-4]+'.h5')
        pl2_to_h5(file,pl2_dir,min_licks)
