# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
import os
import shutil
import sys
sys.path.insert(1, '/data/home/dmarshal/Autosort/pypl2')
import Autosorting as AS
import config_handler
import time
import multiprocessing   # Used for multiprocessing to speed up analyses
import math              # Used to round values
import tables            # Used to read the hdf5 files
import traceback
import warnings

if __name__ == '__main__':
    params=config_handler.do_the_config()
    print('Running Script: '+__file__)
    try:
        n_files=int(params['n-files'])
        to_run_path=params['h5 to-run path'] #path where files to run on were placed
        running_path=params['running path'] #path where files will be stored while the script is running on them
        outpath=params['results path'] #path where output will be generated
        num_cpu=int(params['cores used'])
        resort_limit=int(params['resort limit'])
        filedates=[]
        runfiles=[]
        checkfiles=os.listdir(to_run_path) #get the names of the files to be run
        iterfiles=checkfiles.copy() #filelist for iteration
        for i in range(0,len(iterfiles)):
            if not iterfiles[i].endswith('.h5'):
                checkfiles.remove(iterfiles[i]) #remove everything other than h5 from the list
                continue
            filedates.append(os.path.getctime(to_run_path+'/'+iterfiles[i])) #get the creation date for the file
        if len(filedates)==0:
            sys.exit("No files to run")
        for i in range(0,n_files):#This loop gets the most recent files, up to 10 of them
            if len(filedates)==0: #if there are no more files, end the loop
                break
            runfiles.append(checkfiles[filedates.index(min(filedates))])
            checkfiles.remove(checkfiles[filedates.index(min(filedates))])
            filedates.remove(min(filedates))
        for file in runfiles: #move these files to the running path
            shutil.move(to_run_path+'/'+file,running_path)
        
        for file in runfiles:
            filestart=time.time()
            filename=running_path+'/'+file
            hdf5_name = filename
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
            ###################################################################################################
            ################################   Set up multiprocessing   #######################################
            # If no continuous data was recorded, skip the processing step
            runs = math.ceil(elNum/num_cpu) # Determine how many times to loop through processing based on cpu cores used and number of electrodes
            for n in range(runs): # For the number of runs
                a = num_cpu*n # First electrode to start with
                b = num_cpu*(n+1) # Electrode to stop with
                if b>elNum: # If the last electrode is within the group
                    b = elNum # That will be the last electrode to run processing on
                print("Currently analyzing electrodes %i-%i." %(a+1,b))
                processes=[]
                for i in range(a,b): # run so many times 
                    p = multiprocessing.Process(target = AS.Processing, args = (i,filename,params)) # Run Processing function using multiple processes and input argument i
                    p.start() # Start the processes
                    processes.append(p) #catalogue the processes
                for p in processes:
                    p.join()# rejoin the individual processes once they are all finished
            ### Integrity check
            print("Performing integrity check for sort...")
            bad_runs=[9001]
            reruns=0
            while len(bad_runs)>0 and reruns<resort_limit:
                reruns+=1
                bad_runs=[]
                for chan in range(1,elNum+1):
                    if not os.path.isfile(os.path.splitext(filename)[0]+'/clustering_results/electrode {}'.format(str(chan))+'/success.txt'):
                        bad_runs.append(chan)
                if len(bad_runs)==0:
                    print("All channels were sorted successfully!")
                elif len(bad_runs)==1:
                    print('Channel',bad_runs[0],'was not sorted successfully, resorting...')
                    try: AS.Processing(bad_runs[0]-1,filename,params)
                    except: traceback.print_exc()
                else:
                    if len(bad_runs)>=elNum:
                        raise Exception("Sorting failed on every channel. Do not close python, talk to Daniel")
                    elif len(bad_runs)>num_cpu:
                        bad_runs=bad_runs[0:num_cpu]
                    print('The following channels:',bad_runs, 'were not sorted successfully. Initializing parallel processing for resort...')
                    reprocesses=[]
                    for channel in bad_runs: # run so many times 
                        re = multiprocessing.Process(target = AS.Processing, args = (channel-1,filename,params)) # Run Processing function using multiple processes and input argument i
                        re.start() # Start the processes
                        reprocesses.append(re) #catalogue the processes
                    for re in reprocesses:
                        re.join()# rejoin the individual processes once they are all finished
            #superplots
            bad_runs=[]
            for chan in range(1,elNum+1): #check again for bad runs
                if not os.path.isfile(os.path.splitext(filename)[0]+'/clustering_results/electrode {}'.format(str(chan))+'/success.txt'):
                    bad_runs.append(chan)
            if len(bad_runs)>0: #If there are bad runs don't make superplots
                warnings.warn("Warning: Sort unsuccessful on at least one channel!")
            else: #else make the superplots
                try: AS.superplots(filename,int(params['max clusters']))
                except Exception as e: 
                    warnings.warn("Warning: superplots unsuccessful!")
                    print(e)
                try: AS.compile_isoi(filename,int(params['max clusters']),Lrat_cutoff=float(params['l-ratio cutoff']))
                except Exception as e: 
                    warnings.warn("Warning: isolation information compilation unsuccessful!")
                    print(e)
            sort_time=time.time()-filestart
            print("That file ran for",(sort_time/3600),"hours.")        
            AS.infofile(file,os.path.splitext(filename)[0],sort_time,__file__,params)
        
    except Exception: #if an error occurs, reset everything and print out the error
        traceback.print_exc()
        for file in runfiles: 
            shutil.move(running_path+'/'+file,'/data/home/dmarshal/Autosort/error')
            if os.path.isdir(hdf5_name[:-3]):
                shutil.rmtree(hdf5_name[:-3])
        
    else:
        for file in runfiles: #move the results to the results folder
            shutil.move(running_path+'/'+os.path.splitext(file)[0]+'.h5',outpath+'/'+os.path.splitext(file)[0]+'.h5')
            shutil.move(running_path+'/'+os.path.splitext(file)[0],outpath+'/'+os.path.splitext(file)[0].replace(" ",'_'))
        
    print("done")
        