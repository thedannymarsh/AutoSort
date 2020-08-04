# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:03:55 2020

@author: Di Lorenzo Tech
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 10:08:58 2020

@author: Di Lorenzo Tech
"""
import pandas as pd
import os
import shutil
import sys
from pypl2.pypl2api import pl2_ad, pl2_events, pl2_info # Used to obtain info from Pl2 files
import tables            # Used to read the hdf5 files
import pypl2.Clustering as clust # Used to perform clustering analysis
from scipy.spatial.distance import mahalanobis  # Used to get distance between clusters
from scipy import linalg                # Used for very fast linear algebra
import pypl2.Pl2_waveforms_datashader   # Used for waveform collection
import numpy as np            # Used for number processing
import pylab as plt           # Used as a combo between pyplot and numpy
import matplotlib.cm as cm    # Used for plots
import matplotlib             # Used for plots
import cv2
from PIL import ImageFont, ImageDraw, Image
matplotlib.use('Agg')
import configparser
from datetime import date
import traceback
import warnings
import time



def infofile(pl2_filename,path,sort_time,AS_file,params):
    #dumps run info to a .info file
    config = configparser.ConfigParser()
    config['METADATA']={'Pl2 File':pl2_filename, 'Run Time':sort_time, 'Creator Script': AS_file,'Run Date':date.today().strftime("%m/%d/%y")}
    config['PARAMS USED']=params
    with open(path+'/'+os.path.splitext(pl2_filename)[0]+'_'+'sort.info','w') as infofile:
        config.write(infofile)

 
def pl2_to_h5(filename,filedir,min_licks=1000):
    
    ##################  Functions  ######################
    # Getting continuous data from pl2 file and creating folders
    def getSPKC(dig, name):
        spkcNames = list(set(a.name for a in adinfo if(a.name[:dig] == name)))  # get names of continuous data and add to set list
        spkcNames.sort() # Sort channels in alphabetical order
    

        
        return (spkcNames)
    
    # Run if no continuous data was recored
    def NoContinuous():
        try:
            os.mkdir(filename[:-4])
        except:
            pass
        f= open("%s\\NoCellSortingRunOnThisFile.txt" %filename[:-4],"w+") # Create text file
        f.write("This file has no continuous spike data and so no cell sorting can be run.") # Notify that no continuous data was recorded
        f.close() # Close text file
        
    # Run if less than 1000 licks were recorded
    def NotEnoughLicks():
        try:
            os.mkdir(filename[:-4])
        except:
            pass
        f= open("%s\\NoCellSortingRunOnThisFile.txt" %filename[:-4],"w+") # Create text file
        f.write("The rat did not lick enough and so cell sorting is irrelevant.") # Notify that no continuous data was recorded
        f.close() # Close text file
     
    
    
    #####################################################################################################        
    #################################################  ##################################################
    ##########################################                 ##########################################
    ######################################     PreProcessing      #######################################
    ##########################################                 ##########################################
    #################################################  ##################################################
    #####################################################################################################
    
    
    # Change to directory containing files
    try:
        os.chdir(filedir)
    except:
        sys.exit() # Exit program
  
    ###########################################################################################
    ####################################  Obtain Data  ########################################
    filename = filedir + '\\' + filename
    print("Working on file: %s" % filename)
    
    if os.path.exists(filename[:-4] + ".h5") == False: # Run preprocessing step only if it has not already been done
        #Get file info
        os.chdir("C:\ProgramData\Anaconda3\Lib\site-packages\pypl2") # Must change directory to where the dll files are stored
        spkinfo, evtinfo, adinfo = pl2_info(filename)
        del spkinfo # Delete unnecessary variables


        ##########   Event Data   ##########    
        #Get event data on select channels and print out interesting information
        print("\nEvent Data from pl2_events()" \
              "\nEvent          Number of Events"\
              "\n-------------- ----------------")
    
        evtNames = [0]*len(evtinfo) # Used to store event names
        if len(evtNames) == 0: # If there were no events
            NotEnoughLicks() # Run this function
            del evtNames, evtinfo, adinfo # Delete unnecessary variables
            return
        evtTimes = [0]*len(evtinfo) # Used to store event timestamps
        for n in range(len(evtinfo)):
            if type(evtinfo[n].name) is bytes:
                evtNames[n] = str(evtinfo[n].name)[2:-1] # Turn byte type into string and remove byte symble
            else:
                evtNames[n] = evtinfo[n].name
            #currEvt = pl2_events(filename, evtinfo[n].channel)
            currEvt = pl2_events(filename, evtNames[n])

            if n >1: # for the second event and beyond
                if evtinfo[n].name == evtinfo[n-1].name: # if current event name  is the same as the last event name
                     evtNames[n]= evtNames[n]+" Dup2"
                     evtNames[n-1]= evtNames[n-1]+" Dup1"
                     currEvt = pl2_events(filename, evtinfo[7].channel, n)

            print("{:<15} {:<16}".format(evtNames[n], currEvt.n,))
            
            evtTimes[n]= list(set(a for a in currEvt.timestamps))
            evtTimes[n].sort()
            if (evtNames[n] == 'Lick') | (evtNames[n] == 'lick'): # When events gets to lick event
                l = n # Identify which event is lick
        try:
            if len(evtTimes[l]) < min_licks: # Check and see if there are enough licks
                NotEnoughLicks() # Run this function
                del evtinfo, currEvt, evtNames, evtTimes, adinfo # Delete unnecessary variables
                return
        except:
            pass

        del evtinfo, currEvt, l # Delete unnecessary variables


        ##################  Set Up HDF5 File and Save Events  ######################
        # Initiallize dhf5 file
        os.chdir(filedir)
        hdf5_name = str.split(filename, '\\')
        # Create hdf5 file and make groups for raw spkc data and digital inputs
        hf5 = tables.open_file(filename[:-4]+ '.h5', 'w', title = hdf5_name[-1][:-4])
        hf5.create_group('/', 'filename', filename)
        hf5.create_group('/', 'events')
        hf5.create_group('/', 'SPKC')
        hf5.create_group('/', 'SpikeTimes')
        hf5.create_group('/', 'SpikeValues')
    
        # Input event and spkc arrays into file and close
        print("Currently saving events.")
        for n in range(len(evtNames)):
            hf5.create_array('/events', evtNames[n], evtTimes[n]) # Save event data
        del evtNames, evtTimes  # Delete unnecessary variables
        hf5.close()

        ##########   Continuous Data   ##########
        # Get all continuous spike (SPKC) channels from continuous channels
        # Depending on various situations, recordings can have different numbers of continuous channels
        # We only want SPKC channels if it is present. WB or AD data will be acceptable if SPKC is not
        
        # Determine format and convert to string format if necessary
        os.chdir("C:\ProgramData\Anaconda3\Lib\site-packages\pypl2") # Must change directory to where the dll files are stored
        spkc = False # Initially set spkc data to false
        wb = False # Initially set wb data to false
        ad = False # Initially set ad data to false
        con = False # Initially set continuous data to false
        if type(adinfo[0][1]) is bytes: # If channel name is in byte format
            for n in range(len(adinfo)): # Determine if continuous data were collected
                if adinfo[n][1][:4] == b'SPKC': # If spkc data
                    if adinfo[n][2]>0: # If spkc was recorded
                        spkc = True # Set to true
                        con = True # Set to true
                elif adinfo[n][1][:2] == b'WB': # If wb data
                    if adinfo[n][2]>0: # If wb was recorded
                        wb = True # Set to true
                        con = True # Set to true
                elif adinfo[n][1][:2] == b'AD': # If this was originally a plx file and has AD data
                    if adinfo[n][2]>0: # If AD was recorded
                        ad = True # Set to true
                        con = True # Set to true
            if spkc == True: # If spkc was recorded
                spkcNames = getSPKC(4, b'SPKC') # Add to set list
            elif wb == True: # If no spkc was recorded but wideband (wb) was recorded
                spkcNames = getSPKC(2, b'WB') # Add to set list
            elif ad == True: # If originally plx and analog-to-digital (ad) data was recorded
                spkcNames = getSPKC(2, b'AD') # Add to set list  
            else: # If no spkc or wb was recorded
                NoContinuous()
                del adinfo
                return
                #spkinfo, evtinfo, adinfo = pl2_info(filename)
                #del evtinfo, adinfo # Delete unnecessary variables
                #spNames = getSP() # Add to set list
            if con == True:
                for n in range(len(spkcNames)): # For each spkc channel
                    spkcNames[n] = str(spkcNames[n])[2:-1] # Convert to string format
        else: # if channel name is not in byte format
            for n in range(len(adinfo)): # Determine if continuous data were collected
                if adinfo[n][1][:4] == 'SPKC': # If spkc data
                    if adinfo[n][2]>0: # If spkc was recorded
                        spkc = True # Set to true
                        con = True # Set to true
                elif adinfo[n][1][:2] == 'WB': # If wb data
                    if adinfo[n][2]>0: # If wb was recorded
                        wb = True # Set to true
                        con = True # Set to true
                elif adinfo[n][1][:2] == 'AD': # If this was originally a plx file and has AD data
                    if adinfo[n][2]>0: # If AD was recorded
                        ad = True # Set to true
                        con = True # Set to true
            if spkc == True: # If spkc was recorded
                spkcNames = getSPKC(4, 'SPKC') # Add to set list
            elif wb == True: # If no spkc was recorded but wideband (wb) was recorded
                spkcNames = getSPKC(2, 'WB') # Add to set list 
            elif ad == True: # If originally plx and ad data was recorded
                spkcNames = getSPKC(2, 'AD') # Add to set list  
            else: # If no spkc or wb was recorded
                NoContinuous()
                del adinfo
        del spkc, wb, ad
        
        if con == True:
            print("\nContinuous A/D Channel Info from pl2_info()"\
                  "\nChannel Name    Frequency   Count"\
                  "\n-------------  ----------- ----------")
                        # Display all SPKC channels and values. This will take a little while for each channel
            for n in range(len(spkcNames)):
                os.chdir("C:\ProgramData\Anaconda3\Lib\site-packages\pypl2") # Must change directory to where the dll files are stored
                currSpkc = pl2_ad(filename, spkcNames[n])
                spkcValues = list(a*1000000 for a in currSpkc.ad)
                print("{:<15} {:<11} {}".format(spkcNames[n], int(currSpkc.adfrequency),  currSpkc.n))

                ##################  Save Continuous to HDF5 File  ######################
                # This step is performed after each channel is obtained to limit amount of RAM used
                os.chdir(filedir) # Change directory
                hf5 = tables.open_file(filename[:-4]+ '.h5', 'r+') # Open dhf5 file
                hf5.create_array('/SPKC', 'SPKC%02d' % n, spkcValues) # Save spkc data
                hf5.close()
            del adinfo, currSpkc, spkcValues, hdf5_name, n # Delete unnecessary variables
        else:
            sys.exit("Tell Steve he should have recorded continuous data...")

    else: # If the h5 file has already been created
        print("h5 file already created, skipping that step")


    #####################  Dump Everything And Start The Processing  #############################
    # Make a directory for dumping files talking about memory usage in Pl2_processing.py
    print("Currently dumping files.")


def Processing(electrode_num,pl2_fullpath, params): # Define function
    
    
    #########################################################################################################
    ###########################################   Pl2_Processing   ##########################################
    
    # This script is used to read hdf5 files and identify possible cells. The cells are into different numbers
    # of clusters of which the user will identify which set of clusters to use and which clusters are cells
    # This script is called by Pl2_PreProcessing and should be placed with other modules used by python.
    
    #################    Define the function that will do the actual processing   #################
    retried=0
    while True:
        try:
            filename=os.path.splitext(pl2_fullpath)[0]+'\n'
            #print(filename)
         
            filedir=[os.path.split(pl2_fullpath)[0]+'\n']
            
            # Change directory to the folder containing the hdf5 files
            os.chdir(filedir[0][:-1])
        
            # find the hdf5 (.h5) file
            hdf5_name = filename[:-1] + '.h5'
            #print("Opening file " + hdf5_name)
            
        
            
            #print("Analyzing electrode number " + str(electrode_num+1))
            # Check if the directories for this electrode number exist - if they do, delete them (existence of the directories indicates a job restart on the cluster, so restart afresh)
            if os.path.isdir(hdf5_name[:-3] +'/Plots/'+str(electrode_num+1)):
                shutil.rmtree(hdf5_name[:-3] +'/Plots/'+str(electrode_num+1))
            if os.path.isdir(hdf5_name[:-3] +'/spike_waveforms/electrode '+str(electrode_num+1)):
                shutil.rmtree(hdf5_name[:-3] +'/spike_waveforms/electrode '+str(electrode_num+1))
            if os.path.isdir(hdf5_name[:-3] +'/spike_times/electrode '+str(electrode_num+1)):
                shutil.rmtree(hdf5_name[:-3] +'/spike_times/electrode '+str(electrode_num+1))
            if os.path.isdir(hdf5_name[:-3] +'/clustering_results/electrode '+str(electrode_num+1)):
                shutil.rmtree(hdf5_name[:-3] +'/clustering_results/electrode '+str(electrode_num+1))
            
            # Then make all these directories
            os.mkdir(hdf5_name[:-3] +'/Plots/'+str(electrode_num+1))
            os.mkdir(hdf5_name[:-3] +'/spike_waveforms/electrode '+str(electrode_num+1))
            os.mkdir(hdf5_name[:-3] +'/spike_times/electrode '+str(electrode_num+1))
            os.mkdir(hdf5_name[:-3] +'/clustering_results/electrode '+str(electrode_num+1))
            
            # Assign the parameters to variables
            max_clusters = int(params['max clusters'])
            num_iter = int(params['max iterations'])
            thresh = float(params['convergence criterion'])
            num_restarts = int(params['random restarts'])
            voltage_cutoff = float(params['disconnect voltage'])
            max_breach_rate = float(params['max breach rate'])
            max_secs_above_cutoff = int(params['max breach count'])
            max_mean_breach_rate_persec = float(params['max breach avg.'])
            wf_amplitude_sd_cutoff = float(params['intra-cluster cutoff'])
            bandpass_lower_cutoff = float(params['low cutoff'])
            bandpass_upper_cutoff = float(params['high cutoff'])
            spike_snapshot_before = float(params['pre-time'])
            spike_snapshot_after = float(params['post-time'])
            sampling_rate = float(params['sampling rate'])
            STD=float(params['spike detection'])
            cutoff_std=float(params['artifact removal'])
            pvar=float(params['variance explained'])
            usepvar=int(params['use percent variance'])
            userpc=int(params['principal component n'])
            min_L=float(params['l-ratio cutoff'])
            
            
            # Open up hdf5 file, and load this electrode number
            hf5 = tables.open_file(hdf5_name, 'r')
            spkc=getattr(hf5.root.SPKC,'SPKC'+f'{electrode_num:02d}')[:]
            hf5.close()
            
            # High bandpass filter the raw electrode recordings
            filt_el = clust.get_filtered_electrode(spkc, freq = [bandpass_lower_cutoff, bandpass_upper_cutoff], sampling_rate = sampling_rate)
            
            # Delete raw electrode recording from memory
            del spkc
            
            # Calculate the 3 voltage parameters
            breach_rate = float(len(np.where(filt_el>voltage_cutoff)[0])*int(sampling_rate))/len(filt_el)
            test_el = np.reshape(filt_el[:int(sampling_rate)*int(len(filt_el)/sampling_rate)], (-1, int(sampling_rate)))
            breaches_per_sec = [len(np.where(test_el[i] > voltage_cutoff)[0]) for i in range(len(test_el))]
            breaches_per_sec = np.array(breaches_per_sec)
            secs_above_cutoff = len(np.where(breaches_per_sec > 0)[0])
            if secs_above_cutoff == 0:
                mean_breach_rate_persec = 0
            else:
                mean_breach_rate_persec = np.mean(breaches_per_sec[np.where(breaches_per_sec > 0)[0]])
            
            # And if they all exceed the cutoffs, assume that the headstage fell off mid-experiment
            recording_cutoff = int(len(filt_el)/sampling_rate)
            if breach_rate >= max_breach_rate and secs_above_cutoff >= max_secs_above_cutoff and mean_breach_rate_persec >= max_mean_breach_rate_persec:
                # Find the first 1 second epoch where the number of cutoff breaches is higher than the maximum allowed mean breach rate 
                recording_cutoff = np.where(breaches_per_sec > max_mean_breach_rate_persec)[0][0]
            
            # Dump a plot showing where the recording was cut off at
            fig = plt.figure()
            plt.plot(np.arange(test_el.shape[0]), np.mean(test_el, axis = 1))
            plt.plot((recording_cutoff, recording_cutoff), (np.min(np.mean(test_el, axis = 1)), np.max(np.mean(test_el, axis = 1))), 'k-', linewidth = 4.0)
            plt.xlabel('Recording time (secs)')
            plt.ylabel('Average voltage recorded per sec (microvolts)')
            plt.title('Recording cutoff time (indicated by the black horizontal line)')
            fig.savefig(hdf5_name[:-3] +'/Plots/%i/cutoff_time.png' % (electrode_num+1), bbox_inches='tight')
            plt.close("all")
            
            # Then cut the recording accordingly
            filt_el = filt_el[:recording_cutoff*int(sampling_rate)]    
            
            # Slice waveforms out of the filtered electrode recordings
            slices, spike_times = clust.extract_waveforms(filt_el, spike_snapshot = [spike_snapshot_before, spike_snapshot_after], sampling_rate = sampling_rate, STD = STD,cutoff_std=cutoff_std)
            
            # Delete filtered electrode from memory
            del filt_el, test_el
            
            break
        except MemoryError:
            if retried==1:
                traceback.print_exc()
                return
            warnings.warn("Warning, could not allocate memory for electrode {}".format(electrode_num+1))
            retried=1
            time.sleep(300)
        except: 
            traceback.print_exc()
            return
    
    if len(slices)==0 or len(spike_times)==0:
        with open(hdf5_name[:-3] +'/Plots/'+str(electrode_num+1)+'/'+'no_spikes.txt', 'w') as txt:
            txt.write('No spikes were found on this channel. The most likely cause is an early recording cutoff. RIP')
            
    else:
            
        # Dejitter these spike waveforms, and get their maximum amplitudes
        slices_dejittered, times_dejittered = clust.dejitter(slices, spike_times, spike_snapshot = [spike_snapshot_before, spike_snapshot_after], sampling_rate = sampling_rate)
        amplitudes = np.min(slices_dejittered, axis = 1)
        
        # Delete the original slices and times now that dejittering is complete
        del slices; del spike_times
        
        # Save these slices/spike waveforms and their times to their respective folders
        np.save(hdf5_name[:-3] +'/spike_waveforms/electrode %i/spike_waveforms.npy' % (electrode_num+1), slices_dejittered)
        np.save(hdf5_name[:-3] +'/spike_times/electrode %i/spike_times.npy' % (electrode_num+1), times_dejittered)
        
        # Scale the dejittered slices by the energy of the waveforms
        scaled_slices, energy = clust.scale_waveforms(slices_dejittered)
        
        # Run PCA on the scaled waveforms
        pca_slices, explained_variance_ratio = clust.implement_pca(scaled_slices)
        
        #get cumulative variance explained
        cumulvar=np.cumsum(explained_variance_ratio)
        graphvar=list(cumulvar[0:np.where(cumulvar>.999)[0][0]+1])

        if usepvar==1:n_pc=np.where(cumulvar>pvar)[0][0]+1
        else: n_pc=userpc
        
        #note need to add userpvar, usepvar, and n_pc to params
        # Save the pca_slices, energy and amplitudes to the spike_waveforms folder for this electrode
        np.save(hdf5_name[:-3] +'/spike_waveforms/electrode %i/pca_waveforms.npy' % (electrode_num+1), pca_slices)
        np.save(hdf5_name[:-3] +'/spike_waveforms/electrode %i/energy.npy' % (electrode_num+1), energy)
        np.save(hdf5_name[:-3] +'/spike_waveforms/electrode %i/spike_amplitudes.npy' % (electrode_num+1), amplitudes)
        
        
        # Create file for saving plots, and plot explained variance ratios of the PCA
        fig = plt.figure()
        x = np.arange(0,len(graphvar)+1)
        graphvar.insert(0,0)
        plt.plot(x, graphvar)
        plt.vlines(n_pc,0,1,colors='r')
        plt.annotate(str(n_pc)+" PC's used for GMM.\nVariance explained= " + str(round(cumulvar[n_pc-1],3))+"%.",(n_pc+.25,cumulvar[n_pc-1]-.1))
        plt.title('Variance ratios explained by PCs (cumulative)')
        plt.xlabel('PC #')
        plt.ylabel('Explained variance ratio')
        fig.savefig(hdf5_name[:-3] +'/Plots/%i/pca_variance.png' % (electrode_num+1), bbox_inches='tight')
        plt.close("all")
        
        # Make an array of the data to be used for clustering, and delete pca_slices, scaled_slices, energy and amplitudes
        data = np.zeros((len(pca_slices), n_pc + 2))
        data[:,2:] = pca_slices[:,:n_pc]
        data[:,0] = energy[:]/np.max(energy)
        data[:,1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))
        del pca_slices; del scaled_slices; del energy
                
        # Run GMM, from 3 to max_clusters
        for i in range(max_clusters-2):
            #print("Creating PCA plots.")
            try:
                model, predictions, bic = clust.clusterGMM(data, n_clusters = i+3, n_iter = num_iter, restarts = num_restarts, threshold = thresh)
            except:
                #print "Clustering didn't work - solution with %i clusters most likely didn't converge" % (i+3)
                continue
        
            # Sometimes large amplitude noise waveforms cluster with the spike waveforms because the amplitude has been factored out of the scaled slices.   
            # Run through the clusters and find the waveforms that are more than wf_amplitude_sd_cutoff larger than the cluster mean. Set predictions = -1 
            #at these points so that they aren't picked up by Pl2_PostProcess
            for cluster in range(i+3):
                cluster_points = np.where(predictions[:] == cluster)[0]
                this_cluster = predictions[cluster_points]
                cluster_amplitudes = amplitudes[cluster_points]
                cluster_amplitude_mean = np.mean(cluster_amplitudes)
                cluster_amplitude_sd = np.std(cluster_amplitudes)
                reject_wf = np.where(cluster_amplitudes <= cluster_amplitude_mean - wf_amplitude_sd_cutoff*cluster_amplitude_sd)[0]
                this_cluster[reject_wf] = -1
                predictions[cluster_points] = this_cluster      
        
            # Make folder for results of i+2 clusters, and store results there
            os.mkdir(hdf5_name[:-3] +'/clustering_results/electrode %i/clusters%i' % ((electrode_num+1), i+3))
            np.save(hdf5_name[:-3] +'/clustering_results/electrode %i/clusters%i/predictions.npy' % ((electrode_num+1), i+3), predictions)
            np.save(hdf5_name[:-3] +'/clustering_results/electrode %i/clusters%i/bic.npy' % ((electrode_num+1), i+3), bic)
        
            # Plot the graphs, for this set of clusters, in the directory made for this electrode
            os.mkdir(hdf5_name[:-3] +'/Plots/%i/%i_clusters' % ((electrode_num+1), i+3))
            colors = cm.rainbow(np.linspace(0, 1, i+3))
        
            for feature1 in range(len(data[0])):
                for feature2 in range(len(data[0])):
                    if feature1 < feature2:
                        fig = plt.figure()
                        plt_names = []
                        for cluster in range(i+3):
                            plot_data = np.where(predictions[:] == cluster)[0]
                            plt_names.append(plt.scatter(data[plot_data, feature1], data[plot_data, feature2], color = colors[cluster], s = 0.8))
                                                
                        plt.xlabel("Feature %i" % feature1)
                        plt.ylabel("Feature %i" % feature2)
                        # Produce figure legend
                        plt.legend(tuple(plt_names), tuple("Cluster %i" % cluster for cluster in range(i+3)), scatterpoints = 1, loc = 'lower left', ncol = 3, fontsize = 8)
                        plt.title("%i clusters" % (i+3))
                        fig.savefig(hdf5_name[:-3] +'/Plots/%i/%i_clusters/feature%ivs%i.png' % ((electrode_num+1), i+3, feature2, feature1))
                        plt.close("all")
        
            for cluster in range(i+3):
                fig = plt.figure()
                cluster_points = np.where(predictions[:] == cluster)[0]
                for other_cluster in range(i+3):
                    mahalanobis_dist = []
                    other_cluster_mean = model.means_[other_cluster, :]
                    other_cluster_covar_I = linalg.inv(model.covariances_[other_cluster, :, :])
                    for points in cluster_points:
                         mahalanobis_dist.append(mahalanobis(data[points, :], other_cluster_mean, other_cluster_covar_I))
                    # Plot histogram of Mahalanobis distances
                    y,binEdges=np.histogram(mahalanobis_dist)
                    bincenters = 0.5*(binEdges[1:] + binEdges[:-1])
                    plt.plot(bincenters, y, label = 'Dist from cluster %i' % other_cluster)    
                    
                plt.xlabel('Mahalanobis distance')
                plt.ylabel('Frequency')
                plt.legend(loc = 'upper right', fontsize = 8)
                plt.title('Mahalanobis distance of Cluster %i from all other clusters' % cluster)
                fig.savefig(hdf5_name[:-3] +'/Plots/%i/%i_clusters/OLD_Mahalonobis_cluster%i.png' % ((electrode_num+1), i+3, cluster))
                plt.close("all")
                
                #correct mahalanobis calculations
            for ref_cluster in range(i+3):
                fig = plt.figure()
                ref_mean = model.means_[ref_cluster, :]
                ref_covar_I = linalg.inv(model.covariances_[ref_cluster, :, :])
                for other_cluster in range(i+3):
                    mahalanobis_dist = []
                    other_cluster_points = np.where(predictions[:] == other_cluster)[0]
                    for point in other_cluster_points:
                          mahalanobis_dist.append(mahalanobis(data[point, :], ref_mean, ref_covar_I))
                    # Plot histogram of Mahalanobis distances
                    y,binEdges=np.histogram(mahalanobis_dist)
                    bincenters = 0.5*(binEdges[1:] + binEdges[:-1])
                    plt.plot(bincenters, y, label = 'Dist from cluster %i' % other_cluster)    
                plt.xlabel('Mahalanobis distance')
                plt.ylabel('Frequency')
                plt.legend(loc = 'upper right', fontsize = 8)
                plt.title('Mahalanobis distance of all clusters from Reference Cluster: %i' % ref_cluster)
                fig.savefig(hdf5_name[:-3] +'/Plots/%i/%i_clusters/Mahalonobis_cluster%i.png' % ((electrode_num+1), i+3, ref_cluster))
                plt.close("all")
            
            # Create file, and plot spike waveforms for the different clusters. Plot 10 times downsampled dejittered/smoothed waveforms.
            # Additionally plot the ISI distribution of each cluster 
            os.mkdir(hdf5_name[:-3] +'/Plots/%i/%i_clusters_waveforms_ISIs' % ((electrode_num+1), i+3))
            x = np.arange(len(slices_dejittered[0])/10) + 1
            ISIList=[]
            for cluster in range(i+3):
                cluster_points = np.where(predictions[:] == cluster)[0]
                fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(slices_dejittered[cluster_points, :], x, dir_name =  hdf5_name[:-3]+"datashader_temp_el" + str(electrode_num+1))
                ax.set_xlabel('Sample ({:d} samples per ms)'.format(int(sampling_rate/1000)))
                ax.set_ylabel('Voltage (microvolts)')
                ax.set_title('Cluster%i' % cluster)
                plt.annotate('wf: '+str(len(np.where(predictions[:] == cluster)[0])),(.14,.85),xycoords='figure fraction')
                fig.savefig(hdf5_name[:-3] +'/Plots/%i/%i_clusters_waveforms_ISIs/Cluster%i_waveforms' % ((electrode_num+1), i+3, cluster))
                plt.close("all")
                
                fig = plt.figure()
                cluster_times = times_dejittered[cluster_points]
                ISIs = np.ediff1d(np.sort(cluster_times))
                ISIs = ISIs/40.0
                plt.hist(ISIs, bins = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, np.max(ISIs)])
                plt.xlim([0.0, 10.0])
                plt.title("2ms ISI violations = %.1f percent (%i/%i)" %((float(len(np.where(ISIs < 2.0)[0]))/float(len(cluster_times)))*100.0, len(np.where(ISIs < 2.0)[0]), len(cluster_times)) + '\n' + "1ms ISI violations = %.1f percent (%i/%i)" %((float(len(np.where(ISIs < 1.0)[0]))/float(len(cluster_times)))*100.0, len(np.where(ISIs < 1.0)[0]), len(cluster_times)))
                fig.savefig(hdf5_name[:-3] +'/Plots/%i/%i_clusters_waveforms_ISIs/Cluster%i_ISIs' % ((electrode_num+1), i+3, cluster))
                plt.close("all") 
                ISIList.append("%.1f" %((float(len(np.where(ISIs < 1.0)[0]))/float(len(cluster_times)))*100.0)  )          
            
            #Get isolation statistics for each solution
            if np.any(np.array([float(x) for x in ISIList])<1):
                isodf=clust.isoinfo(data,predictions,Lrat_cutoff=min_L,isodir=hdf5_name[:-3]+"_temp_isoi_el_" + str(electrode_num+1))
            else: 
                isodf=pd.DataFrame(columns=['IsoIBG','IsoINN','NNClust','IsoD','L-Ratio'],index=range(i+3))
            isodf.insert(1,'ISIs (%)',ISIList) #need to gather ISI's into list
            isodf.insert(0,'wf count',[len(np.where(predictions[:] == cluster)[0]) for cluster in range(i+3)])
            isodf.insert(0,'Solution',i+3) 
            isodf.insert(0,'Channel',electrode_num+1) 
            isodf.insert(0,'File',hdf5_name[:-3]) 
            isodf.insert(0,'IsoRating','TBD') 
            isodf=isodf.round({'IsoIBG':3,'IsoINN':3,'L-Ratio':3,'IsoD':3,})
            isodf.to_csv(hdf5_name[:-3] +'/clustering_results/electrode {}/clusters{}/isoinfo.csv'.format(electrode_num+1, i+3),index=False)

            #output this all in a plot in the plots folder and replace the ISI plot in superplots
            for cluster in range(i+3):
                text='1 ms ISIs (%): \nIsoI-BG: \nIsoI-NN: \nNNClust: \nIsoD: \nL-Ratio: ' #package text to be plotted
                text2='{}\n{}\n{}\n{}\n{}\n{}'.format(isodf['ISIs (%)'][cluster], isodf['IsoIBG'][cluster],isodf['IsoINN'][cluster],
                                                      isodf['NNClust'][cluster],isodf['IsoD'][cluster],isodf['L-Ratio'][cluster])
                blank=np.ones((480,640,3),np.uint8)*255 #initialize empty whihte image
                cv2_im_rgb=cv2.cvtColor(blank,cv2.COLOR_BGR2RGB)   #convert to color space pillow can use
                pil_im=Image.fromarray(cv2_im_rgb)  #get pillow image
                draw=ImageDraw.Draw(pil_im)   #create draw object for text
                font=ImageFont.truetype(os.path.split(__file__)[0]+"/bin/arial.ttf", 60)  #use arial font
                draw.multiline_text((35, 20), text, font=font,fill=(0,0,0,255),spacing=20,align='right')   #draw the text
                draw.multiline_text((420, 20), text2, font=font,fill=(0,0,0,255),spacing=20)   #draw the text
                isoimg=cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  #convert back to openCV image
                cv2.imwrite(hdf5_name[:-3] +'/Plots/{}/{}_clusters_waveforms_ISIs/Cluster{}_Isolation.png'.format(electrode_num+1, i+3, cluster),isoimg) #save the image
            
def superplots(full_filename,maxclust):
    #This function takes all of the plots and conglomerates them
    path=os.path.splitext(full_filename)[0]+'/Plots'  #The path holding the plots to be run on
    outpath=os.path.splitext(full_filename)[0]+'/superplots' #output path for superplots
    if os.path.isdir(outpath): #if the path for plots exists remove it
        shutil.rmtree(outpath)
    os.mkdir(outpath) #make the output path
    for channel in os.listdir(path): #for each channel
        try:
            currentpath=path+'/'+channel 
            os.mkdir(outpath+'/'+channel) #create an output path for each channel
            for soln in range(3,maxclust+1): #for each number cluster solution
                finalpath=outpath+'/'+channel+'/'+str(soln)+'_clusters'
                os.mkdir(finalpath) #create output folders
                for cluster in range(0,soln): #for each cluster
                    mah=cv2.imread(currentpath+'/'+str(soln)+'_clusters/Mahalonobis_cluster'+str(cluster)+'.png')
                    if not np.shape(mah)[0:2]==(480,640):
                        mah=cv2.resize(mah,(640,480))
                    wf=cv2.imread(currentpath+'/'+str(soln)+'_clusters_waveforms_ISIs/Cluster'+str(cluster)+'_waveforms.png')
                    if not np.shape(mah)[0:2]==(1200,1600):
                        wf=cv2.resize(wf,(1600,1200))
                    isi=cv2.imread(currentpath+'/'+str(soln)+'_clusters_waveforms_ISIs/Cluster'+str(cluster)+'_Isolation.png')
                    if not np.shape(isi)[0:2]==(480,640):
                        isi=cv2.resize(isi,(640,480))
                    blank=np.ones((240,640,3),np.uint8)*255 #make whitespace for info
                    text="Electrode: "+channel+"\nSolution: "+str(soln)+"\nCluster: "+str(cluster) #text to output to whitespace (cluster, electrode, and solution numbers)
                    cv2_im_rgb=cv2.cvtColor(blank,cv2.COLOR_BGR2RGB)   #convert to color space pillow can use
                    pil_im=Image.fromarray(cv2_im_rgb)  #get pillow image
                    draw=ImageDraw.Draw(pil_im)   #create draw object for text
                    font=ImageFont.truetype(os.path.split(__file__)[0]+"/bin/arial.ttf", 60)  #use arial font
                    draw.multiline_text((170, 40), text, font=font,fill=(0,0,0,255),spacing=10)   #draw the text
                    info=cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  #convert back to openCV image
                    im_v=cv2.vconcat([info,mah,isi]) #concatenate images together
                    im_all=cv2.hconcat([wf,im_v]) #continued concatenation
                    cv2.imwrite(finalpath+'/Cluster_'+str(cluster)+'.png',im_all) #save the image
        except Exception as e:
            print("Could not create superplots for channel " +channel+ ". Encountered the following error: "+e)
            

def compile_isoi(full_filename,maxclust):
    #compiles all isolation information into one excel file
    path=os.path.splitext(full_filename)[0]+'/clustering_results'
    file_isoi=pd.DataFrame()
    errorfiles=pd.DataFrame(columns=['channel','solution','file'])
    for channel in os.listdir(path): #for each channel
        channel_isoi=pd.DataFrame()
        for soln in range(3,maxclust+1): #for each solution
            try: #get the isoi info for this solution and add it to the channel data
                channel_isoi=channel_isoi.append(pd.read_csv(path +'/{}/clusters{}/isoinfo.csv'.format(channel, soln)))
            except Exception as e: #if an error occurs, add it to the list of error files
                print(e)
                errorfiles=errorfiles.append([{'channel':channel[-1],'solution':soln,'file':os.path.split(path)[0]}])
        channel_isoi.to_csv('{}/{}/{}_iso_info.csv'.format(path,channel,channel),index=False) #output data for the whole channel to the proper folder
        file_isoi=file_isoi.append(channel_isoi) #add this channel's info to the whole file info
    with pd.ExcelWriter(os.path.split(path)[0]+'/{}_compiled_isoi.xlsx'.format(os.path.split(path)[-1]),engine='xlsxwriter') as outwrite:
        #once the file data is compiled, write to to an excel file
        file_isoi.to_excel(outwrite,sheet_name='iso_data',index=False)
        if errorfiles.size==0: #if there are no error csv's add some nans and output to the excel
            errorfiles=errorfiles.append([{'channel':'nan','solution':'nan','file':'nan'}])
        errorfiles.to_excel(outwrite,sheet_name='errors')
        workbook  = outwrite.book
        worksheet = outwrite.sheets['iso_data']
        redden= workbook.add_format({'bg_color':'red'})
        yellen = workbook.add_format({'bg_color':'yellow'})
        #add conditional formatting based on ISI's
        worksheet.conditional_format('A2:L{}'.format(file_isoi.shape[0]+1),{'type':'formula','criteria':'=$G2>1','format':redden}) #new formula for when we have established number: =OR($F2>.5,$K2>{}).format(Lrat_cutoff)) Note, will need to change lrat to isoiNN and BG
        worksheet.conditional_format('A2:L{}'.format(file_isoi.shape[0]+1),{'type':'formula','criteria':'=$G2>.5','format':yellen}) #new formula for when we have established number: =OR($F2>.5,$K2>{}).format(Lrat_cutoff)) Note, will need to change lrat to isoiNN and BG
        outwrite.save() #need to get the correct ISI column here
        