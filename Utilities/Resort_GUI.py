# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
import os
import pypl2.Autosorting as AS
import pypl2.config_handler
import time
import tables
import warnings
import easygui
import traceback

path=r'R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted' #place your file containing path here if usepaths=0

if __name__ == '__main__':
    num_cpu=1
    print('Running Script: '+__file__)
    params=pypl2.config_handler.do_the_config()   
    min_licks=int(params['minimum licks'])
    rerun={}
    runfiles=[file for file in os.listdir(path) if file.endswith('.pl2')]
    for file in runfiles:
        rerun[file]=easygui.multchoicebox(msg='Select channels to rerun for:\n\n'+file,choices=range(1,17))
    for file in runfiles:
        if rerun[file] is None: continue
        start=time.time()
        try:
            AS.pl2_to_h5(file,path,min_licks) #this will pull data from the pl2 file and package it in an h5 file
            filename=path+'/'+file
            hdf5_name = filename[:-4] + '.h5'
            print("Opening file " + hdf5_name)
            hf5 = tables.open_file(hdf5_name, 'r')
            elNum = len(hf5.list_nodes("/SPKC"))
            hf5.close()
            # Create directories to store waveforms, spike times, clustering results, and plots
            if not os.path.isdir(hdf5_name[:-3]):
                os.mkdir(hdf5_name[:-3])
                os.mkdir(hdf5_name[:-3]+'/spike_waveforms')
                os.mkdir(hdf5_name[:-3]+'/spike_times')
                os.mkdir(hdf5_name[:-3]+'/clustering_results')
                os.mkdir(hdf5_name[:-3]+'/Plots')
            #Begin processing
            rerun_channels=rerun[file]
            processed=0
            while processed<len(rerun_channels):
                print("Initializing resort for channel "+rerun_channels[processed]+'...')
                AS.Processing(int(rerun_channels[processed])-1,filename,params)
                processed+=1
            ### Superplots
            try: AS.superplots(filename,int(params['max clusters']))
            except Exception as e: 
                warnings.warn("Warning: superplots unsuccessful!")
                print(e)
            try: AS.compile_isoi(filename,int(params['max clusters']))
            except Exception as e: 
                warnings.warn("Warning: isolation information compilation unsuccessful!")
                print(e)
            sort_time=(time.time()-start)/3600
            print("Resort completed.",filename,"ran for",sort_time,"hours.")
            AS.infofile(file,os.path.splitext(filename)[0],sort_time,__file__,params)
        except Exception:
            print("An error occured while sorting " +filename+". The error is as follows:")
            traceback.print_exc()         
    print("Sorting Complete!")
        