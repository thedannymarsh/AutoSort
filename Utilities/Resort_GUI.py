# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
import os
import pypl2.Autosorting as AS
from pypl2 import config_handler
import time
import warnings
import easygui
import traceback

path=r'D:\VgX\Files\Raw Files VgX\resort' 

if __name__ == '__main__':
    num_cpu=1
    print('Running Script: '+__file__)
    params=config_handler.do_the_config() #get params, or make a file if there is none
    min_licks=int(params['minimum licks'])
    rerun={}
    runfiles=[file for file in os.listdir(path) if file.endswith('.pl2')] #get list of files to run
    for file in runfiles: #for each file, ask the user which channels to resort
        rerun[file]=easygui.multchoicebox(msg='Select channels to rerun for:\n\n'+file,choices=range(1,17))
    for file in runfiles:
        if rerun[file] is None: continue #if the user did not pick and channels, go to the next file
        start=time.time()
        try:
            AS.pl2_to_h5(file,path,min_licks) #this will pull data from the pl2 file and package it in an h5 file
            filename=path+'/'+file
            # Create directories to store waveforms, spike times, clustering results, and plots
            if not os.path.isdir(filename[:-4]): #make empty directories for data
                os.mkdir(filename[:-4])
                os.mkdir(filename[:-4]+'/spike_waveforms')
                os.mkdir(filename[:-4]+'/spike_times')
                os.mkdir(filename[:-4]+'/clustering_results')
                os.mkdir(filename[:-4]+'/Plots')
            #Begin processing
            rerun_channels=rerun[file] #get channels to be rerun
            processed=0
            while processed<len(rerun_channels): #start processing for each channel
                print("Initializing resort for channel "+rerun_channels[processed]+'...')
                AS.Processing(int(rerun_channels[processed])-1,filename,params)
                processed+=1
            ### Superplots
            try: AS.superplots(filename,int(params['max clusters']))
            except Exception as e: 
                warnings.warn("Warning: superplots unsuccessful!")
                print(e)
                ###isoi compiling
            try: AS.compile_isoi(filename,int(params['max clusters']))
            except Exception as e: 
                warnings.warn("Warning: isolation information compilation unsuccessful!")
                print(e)
            sort_time=(time.time()-start)/3600
            print("Resort completed.",filename,"ran for",sort_time,"hours.")
            AS.infofile(file,os.path.splitext(filename)[0],sort_time,__file__,params) #make info file
        except Exception: #if an error occurs tell the user what the error is
            print("An error occured while sorting " +filename+". The error is as follows:")
            traceback.print_exc()         
    print("Sorting Complete!")
        