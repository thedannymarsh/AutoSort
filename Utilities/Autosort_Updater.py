# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 09:58:45 2020

@author: Di Lorenzo Tech
"""
import os
from configparser import ConfigParser
import shutil
import sys
from datetime import date
from importlib import reload

python_path='C:/ProgramData/Anaconda3/Lib/site-packages'
autosort_path=os.path.expanduser('~')+'/Documents/Autosort'
autosort_repo='R:/Daniel/Repositories/Autosort'

def update_autosort():
    #check that the python path is valid
    if not os.path.isdir(python_path):
        sys.exit('Your python path is incorrect')
    
    #replace pypl2 directory
    if os.path.isdir(python_path+'/pypl2'):
        shutil.rmtree(python_path+'/pypl2')
    shutil.copytree(autosort_repo+'/pypl2',python_path+'/pypl2')
    
    #Make the autosort folder if none exists, and copy one time files
    if not os.path.isdir(autosort_path):
        os.mkdir(autosort_path)
        shutil.copy(autosort_repo+'/Json2Nex.py',autosort_path)
        shutil.copy(autosort_repo+'/Utilities/Autosort_Updater.py',autosort_path)
    #update files
    shutil.copy(autosort_repo+'/Autosort_Main.py',autosort_path)
    shutil.copy(autosort_repo+'/Autosort_Post.py',autosort_path)
    
    #make/update config file
    from pypl2 import config_handler
    reload(config_handler)
    config_handler.do_the_config(autosort_path+'/Autosort_config.ini')
    
if os.path.isfile(autosort_path+'/version.info'): #if the version info exists
    config=ConfigParser()
    config.read(autosort_repo+'/version.txt')
    version=config['Autosort']['version'] #get version number
    config.read(autosort_path+'/version.info')
    oldver=config['Autosort']['version'] #get version number installed on this pc
    if oldver!=version: #if this pc's autosort is outdated 
        print('Your Autosort is outdated. Updating...')
        update_autosort()
        config['Autosort']={'version':version,'last update':str(date.today())} #version info
        with open(autosort_path+'/version.info','w') as outfile:
            config.write(outfile) #output version info
        print('Update complete! Version number is '+version)
    else: print('Your Autosort is up to date! Version number is '+version)
else: #otherwise run first time setup
    print('Installing Autosort...')
    update_autosort() #run setup
    config=ConfigParser()
    config.read(autosort_repo+'/version.txt') 
    version=config['Autosort']['version'] #get the current version number
    config['Autosort']={'version':version,'last update':str(date.today())} #version info
    with open(autosort_path+'/version.info','w') as outfile:
        config.write(outfile) #output version info
    print('Installation complete! Version number is '+version)
    print('It is recommended that you create a batch file to activate this script weekly using Windows task scheduler, in order to avoid missing important updates!\nIf you plan on running the automated version of the Autosort, you will need to do the same for the Autosort_main.py script.')
