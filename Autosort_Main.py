# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
import sys
import os
import shutil
import pypl2.Autosorting as AS
import pypl2.config_handler
import time
import multiprocessing   # Used for multiprocessing to speed up analyses
import math              # Used to round values
import tables            # Used to read the hdf5 files
import datetime
import warnings

if __name__ == '__main__':
    params=pypl2.config_handler.do_the_config()
    print('Running Script: '+__file__)
    manual_or_auto=params['run type']
    weekend_n=int(params['weekend run'])
    weekday_n=int(params['weekday run'])
    manual_n=int(params['manual run'])
    usepaths=int(params['use paths'])
    to_run_path=params['pl2 to-run path']
    elsepath=params['else path']
    running_path=params['running path']
    completed_path=params['completed pl2 path']
    outpath=params['results path']
    min_licks=int(params['minimum licks'])
    num_cpu=int(params['cores used'])
    resort_limit=int(params['resort limit'])
    if manual_or_auto=='Auto':
        if datetime.datetime.weekday(datetime.date.today())==4: n_files=weekend_n
        else: n_files=weekday_n
    elif manual_or_auto=='Manual': n_files=manual_n
    else: raise Exception('you only had two choices')
    if usepaths==0:
        to_run_path=elsepath
    filedates=[]
    runfiles=[]
    checkfiles=os.listdir(to_run_path) #get the names of the files to be run
    iterfiles=checkfiles.copy()
    for i in range(0,len(iterfiles)):
        if not iterfiles[i].endswith('.pl2'):
            checkfiles.remove(iterfiles[i])
            continue
        filedates.append(os.path.getctime(to_run_path+'/'+iterfiles[i])) #move the file to the folder for currently running files
    if len(filedates)==0:
        sys.exit("No files to run")
    for i in range(0,n_files):
        if len(filedates)==0: #if there are no more files, end the loop
            break
        runfiles.append(checkfiles[filedates.index(min(filedates))])
        checkfiles.remove(checkfiles[filedates.index(min(filedates))])
        filedates.remove(min(filedates))

    if usepaths==0:
        del to_run_path
    else:
        for file in runfiles:
            shutil.move(to_run_path+'\\'+file,running_path)
    
    ranfiles=runfiles.copy()
    for file in runfiles:
        if usepaths==0:
            running_path=elsepath
        AS.pl2_to_h5(file,running_path,min_licks) #this will pull data from the pl2 file and package it in an h5 file
        if os.path.isfile(running_path+'/'+os.path.splitext(file)[0]+"/NoCellSortingRunOnThisFile.txt"):
            print(file,'cannot be sorted. Probably because there were not enough licks.')
            if usepaths==1:
                shutil.move(running_path+'\\'+file,completed_path+'\\'+file)
                shutil.move(running_path+'\\'+os.path.splitext(file)[0],outpath+'\\'+os.path.splitext(file)[0])
                ranfiles.remove(file)
            continue
        filestart=time.time()
        filename=running_path+'/'+file
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
        elapsed_time=time.time()-filestart
        print("That file ran for",(elapsed_time/3600),"hours.")
        ### Integrity check
        print("Performing integrity check for sort...")
        bad_runs=[9001]
        reruns=0
        while len(bad_runs)>0 and reruns<resort_limit:
            reruns+=1
            bad_runs=[]
            for chan in range(1,elNum+1):
                if not os.path.isfile(os.path.splitext(filename)[0]+'/Plots/'+str(chan)+'/7_clusters_waveforms_ISIs/Cluster6_waveforms.png'):
                    bad_runs.append(chan)
            if len(bad_runs)==0:
                print("All channels were sorted successfully!")
            elif len(bad_runs)==1:
                print('Channel',bad_runs[0],'was not sorted successfully, resorting...')
                AS.Processing(bad_runs[0]-1,filename,params)
            else:
                if len(bad_runs)>=elNum:
                    raise Exception("Sorting failed on every channel. Do not close python, talk to Daniel")
                if len(bad_runs)>num_cpu:
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
            if not os.path.isfile(os.path.splitext(filename)[0]+'/Plots/'+str(chan)+'/7_clusters_waveforms_ISIs/Cluster6_waveforms.png'):
                bad_runs.append(chan)
        if len(bad_runs)>0: #If there are bad runs don't make superplots
            warnings.warn("Warning: Sort unsuccessful on at least one channel!")
        else: #else make the superplots
            try: AS.superplots(filename,params[0])
            except Exception as e: 
                warnings.warn("Warning: superplots unsuccessful!")
                print(e)
        sort_time=str(((time.time()-filestart)/3600))
        print("Sort completed.",filename,"ran for",sort_time,"hours.")
        AS.infofile(file,os.path.splitext(filename)[0],sort_time,__file__,params)
    if usepaths==1:
        for file in ranfiles:
            shutil.move(running_path+'\\'+file,completed_path+'\\'+file)
            shutil.move(running_path+'\\'+os.path.splitext(file)[0]+'.h5',outpath+'\\'+os.path.splitext(file)[0]+'.h5')
            shutil.move(running_path+'\\'+os.path.splitext(file)[0],outpath+'\\'+os.path.splitext(file)[0])
    print("Sorting Complete!")
        