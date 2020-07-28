# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
import os
import pypl2.Autosorting_3_1_2 as AS
import time
import tables
import warnings
import easygui
import traceback

path=r'R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted' #place your file containing path here if usepaths=0

# Clustering parameters: [Maximum number of clusters, Maximum number of iterations, Convergence criterion, Number of random restarts for GMM]
clustering_params = [7, 1000, .0001, 10]
# Data parameters: [Voltage cutoff for disconnected headstage noise (in uV), Maximum rate of cutoff breaches per sec, Maximum number of allowed seconds with at least 1 cutoff breach, 
#                   Maximum allowed average number of cutoff breaches per sec, Intra-cluster waveform amplitude SD cutoff]
data_params = [1500, .2, 10, 20, 3]
# Bandpass parameters: [Lower frequency cutoff (Hz), Upper frequency cutoff (Hz)]
bandpass_params = [300, 6000]
#spike snapshot: [Time before spike minimum (ms) (usually 0.2), Time after spike minimum (ms) (usually 0.6),sampling rate (usually 40kHz=40000)]
spike_snapshot = [.2,.6,40000]
#std_params: [stdev below mean electrode value for detecting a spike, stdev from mean value for eliminating high amplitude artifact]
std_params=[2.0,10.0]
#principal component params: [percent variance to be explained by principal components for use in GMM, 
#                               whether to use percent variance or a flat number of components 9use 1 for percent, 0 otherwise),
#                               number of principal components to use for GMM if not using percent (must enter a value even if not using)]
pca_params=[.95,1,5]

min_licks=1000 #minimum number of licks required to run autosort. If licks are too few the autosort will move on to the next file

### END USER PARAMETERS

if __name__ == '__main__':
    num_cpu=1
    print('Running Script: '+__file__)
    params=clustering_params+data_params+bandpass_params+spike_snapshot+std_params+pca_params
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
            try: AS.superplots(filename,params[0])
            except Exception as e: 
                warnings.warn("Warning: superplots unsuccessful!")
                print(e)
            sort_time=(time.time()-start)/3600
            print("Resort completed.",filename,"ran for",sort_time,"hours.")
            AS.infofile(file,os.path.splitext(filename)[0],sort_time,__file__)
        except Exception:
            print("An error occured while sorting " +filename+". The error is as follows:")
            traceback.print_exc()         
    print("Sorting Complete!")
        