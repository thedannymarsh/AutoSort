# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 15:06:04 2020

@author: Di Lorenzo Tech
"""
#import sys
#sys.path.insert(0, '/Python Packages')

import os
import shutil
from paramiko import SSHClient
from scp import SCPClient
import sys
from pypl2.Autosorting import pl2_to_h5
import contextlib

### USER PARAMETERS

#path with pl2 files to be run
to_run_path='R:\\Dannymarsh Sorting Emporium\\pl2_to_be_sorted' 

#where to store files that are currently running
running_path='R:\\Dannymarsh Sorting Emporium\\running_files_computing_cluster' 

#where to store pl2 files that are finished running
completed_path='R:\\Dannymarsh Sorting Emporium\\completed_pl2' 

#where completed h5 files and plots will be placed
results_path='R:\\Dannymarsh Sorting Emporium\\Results'

#where running logs from the computing cluster will be placed
logs_path='R:\\Dannymarsh Sorting Emporium\\logs\\Cluster'

#where to place the files to run before activating the processing script
remote_to_run_path='/data/home/dmarshal/Autosort/h5_for_run' 

#where the processing script is stored
python_script_path='/data/home/dmarshal/Autosort/'

#location to pull completed files from
remote_results_filepath='/data/home/dmarshal/Autosort/Results'

min_licks=1000

clustrun=5 #cutoff for uploading more files to the cluster

### END USER PARAMETERS

#This function will take a windows path string (make sure it uses double backslash) 
#as an argument and return a unix path as a string
def unix_path(windows_path):
    out=windows_path.replace('\\','/')
    out=out.replace(':','')
    out=out.split('/')
    out[0]=out[0].lower()
    out='/'.join(out)
    out='/mnt/'+out
    return out

### Opening secure connection
print("Establishing secure connection...")
ssh = SSHClient() #open SSH connection
ssh.load_system_host_keys() #get list of known hosts
ssh.connect('spiedie.binghamton.edu',22,username='dmarshal') #connect to specified host and account (username). Shouldn't need a password argument if key authentification is set up properly

# SCPCLient takes a paramiko transport as an argument
scp = SCPClient(ssh.get_transport()) #open secure copy client
sftp=ssh.open_sftp() #open secure file transfer client
###

### Getting completed files

#This loop checks the results directory, and if any h5 files are found it will copy the plots
#folder over to the local directory, and then remove it. It will also move the local h5 file to the 
#results path and the local pl2 file to the completed path
print("Pulling completed data from the server...")
finished_files=sftp.listdir(remote_results_filepath)
for file in finished_files:
    if file.endswith('.h5'):
        print('Downloading completed sort for:', file[:-3]+'...')
        os.system('bash -c "cd /mnt && sudo mount -t drvfs R: r; rsync -r dmarshal@spiedie.binghamton.edu:'+(remote_results_filepath+'/'+os.path.splitext(file)[0]).replace(" ",'_')+' '+unix_path(results_path).replace(" ",'\ ')+'; sudo umount r"')
        i,o,error=ssh.exec_command("rm -r "+"\""+(remote_results_filepath+'/'+os.path.splitext(file)[0]).replace(" ",'_')+"\"")
        shutil.move(running_path+'\\'+os.path.splitext(file)[0]+'.pl2',completed_path+'\\'+os.path.splitext(file)[0]+'.pl2')
        sftp.remove(remote_results_filepath+'/'+file)
        shutil.move(running_path+'\\'+os.path.splitext(file)[0]+'.h5',results_path+'\\'+os.path.splitext(file)[0]+'.h5')
        
#pulling running logs from the server
print('Downloading sorting logs from the server...')
logcheck=sftp.listdir(remote_results_filepath)
for file in logcheck:
  if file.endswith(".log"):
      scp.get(remote_results_filepath+'/'+file,local_path=logs_path)
      sftp.remove(remote_results_filepath+'/'+file)
    
    
### Preparing files to be ran
start_cluster=0
filecount=0
uploaded=0
queue_sbatch=sftp.listdir(remote_to_run_path)

for file in queue_sbatch: #checks to see how many files on the server are waiting to be run
    if file.endswith('.h5'):
        start_cluster+=1

any_files=os.listdir(to_run_path)
for file in any_files: #if there are no pl2 files to run and no h5 files on the cluster waiting to be run, exit
   if file.endswith('.pl2'): 
       any_files='engage'
       filecount+=1
if not any_files=='engage' and start_cluster==0: sys.exit('no files to run')

print("There are", start_cluster, 'uploaded files waiting to run, and',filecount,' files to upload.')

if start_cluster<=clustrun: #if there are not too many files on the cluster
    print("Uploading files to run, and initializing autosort...")
    runfiles=[] 
    checkfiles=os.listdir(to_run_path) #get the names of the files to be run
    iterfiles=checkfiles.copy() #filelist for iteration
    filedates=[]
    for i in range(0,len(iterfiles)):
        if not iterfiles[i].endswith('.pl2'):
            checkfiles.remove(iterfiles[i]) #remove everything other than pl2 from the list
            continue
        filedates.append(os.path.getctime(to_run_path+'/'+iterfiles[i])) #get the creation date for the file
    for i in range(0,len(checkfiles)):#This loop gets the most recent files
        if len(filedates)==0: #if there are no more files, end the loop
            break
        runfiles.append(checkfiles[filedates.index(min(filedates))])
        checkfiles.remove(checkfiles[filedates.index(min(filedates))])
        filedates.remove(min(filedates))
    for file in runfiles: 
        uploaded+=1
        if file.endswith('.pl2'):
            print('Preprocessing', file+'...')
            shutil.move(to_run_path+'\\'+file,running_path+'\\'+file) #move the file to the folder for currently running files
            with open(os.devnull, 'w') as devnull: #suppresses function output
                with contextlib.redirect_stdout(devnull):
                    pl2_to_h5(file,running_path,min_licks) #this will pull data from the pl2 file and package it in an h5 file
            if not os.path.exists((running_path+'\\'+file)[:-4] + "\\NoCellSortingRunOnThisFile.txt"):  # Determine if the rat licked enough
                print('Uploading preprocessed data for',file[:-4]+'...')
                h5_filepath=os.path.splitext(running_path+'\\'+file)[0]+'.h5' 
                os.system('bash -c "cd /mnt && sudo mount -t drvfs R: r; rsync -r'+' '+unix_path(h5_filepath).replace(" ",'\ ')+' '+'dmarshal@spiedie.binghamton.edu:'+(remote_to_run_path)+';sudo umount r"')
                #queue up a job for the autosort on the remote server
                stdin, stdout, stderr = ssh.exec_command("cd "+python_script_path+";module add slurm/current; sbatch start_autosort.sh")
            else: #if sorting cannot be run move the files to finished directories
                print(file,'cannot be sorted.')
                shutil.move(running_path+'\\'+file,completed_path+'\\'+file)
                shutil.move(running_path+'\\'+os.path.splitext(file)[0],results_path+'\\'+os.path.splitext(file)[0])
        if uploaded+start_cluster>=clustrun: #if we've reached the file limit
            break
               
print("File transfers complete. Closing secure connection.")

scp.close()
sftp.close()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
ssh.close()

