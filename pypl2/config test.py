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
import os
import configparser
import easygui

def as_config(path=''):
    if path==''
        if os.name=='nt'
            if not os.path.isfile(os.path.expanduser('~')+'\\Documents\\Autosort_config.txt'):
                pass
        elif os.name=='posix'

def default_config()
    config=configparser.ConfigParser()
    config['Run Settings']={'Resort Limit':'3','Minimum Licks':'1000','Cores Used':'8',
                            'Weekday Run':'2','Weekend Run':'6','Run Type':'Auto','Manual Run':'2'}
    config['Paths']={'Pl2 To-Run Path':r'R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted','Running Path':r'R:\Dannymarsh Sorting Emporium\running_files',
                     'Results Path':r'R:\Dannymarsh Sorting Emporium\Results','Completed Pl2 Path':r'R:\Dannymarsh Sorting Emporium\completed_pl2',
                     'Use Paths':'1','Else Path':''}
    config['Clustering']={'Max Clusters':'7','Max Iterations':'1000','Convergence Criterion':'.0001','Random Restarts':'10'}
    config['Signal']={'Disconnect Voltage':'1500','Max Breach Rate':'.2','Max Breach Count':'10','Max Breach Avg.':'20','Intra-Cluster Cutoff':'3'}
    config['Filtering']={'Low Cutoff':'300', 'High Cutoff':'6000'}
    config['Spike']={'Pre-time':'.2','Post-time':'.6','Sampling Rate':'40000'}
    config['Std Dev']={'Spike Detection':'2.0','Artifact Removal':'10.0'}
    config['PCA']={'Variance Explained':'.95','Use Percent Variance':'1','Principal Component n':'5'}
    with open(os.path.expanduser('~')+'\\Documents\\Autosort.ini','w') as configfile:
        config.write(configfile)
        
def read_config():
    config=configparser.ConfigParser()
    params={}
    config.read(os.path.expanduser('~')+'\\Documents\\Autosort.ini')
    for key,value in config._sections.items():
        params.update(value)
        return params
    
def make_config():
    config=configparser.ConfigParser()
    config=configparser.ConfigParser()
    config['Run Settings']={'Resort Limit':'3','Minimum Licks':'1000','Cores Used':'8',
                            'Weekday Run':'2','Weekend Run':'6','Run Type':'Auto','Manual Run':'2'}
    config['Paths']={'Pl2 To-Run Path':r'R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted','Running Path':r'R:\Dannymarsh Sorting Emporium\running_files',
                     'Results Path':r'R:\Dannymarsh Sorting Emporium\Results','Completed Pl2 Path':r'R:\Dannymarsh Sorting Emporium\completed_pl2',
                     'Use Paths':'1','Else Path':''}
    config['Clustering']={'Max Clusters':'7','Max Iterations':'1000','Convergence Criterion':'.0001','Random Restarts':'10'}
    config['Signal']={'Disconnect Voltage':'1500','Max Breach Rate':'.2','Max Breach Count':'10','Max Breach Avg.':'20','Intra-Cluster Cutoff':'3'}
    config['Filtering']={'Low Cutoff':'300', 'High Cutoff':'6000'}
    config['Spike']={'Pre-time':'.2','Post-time':'.6','Sampling Rate':'40000'}
    config['Std Dev']={'Spike Detection':'2.0','Artifact Removal':'10.0'}
    config['PCA']={'Variance Explained':'.95','Use Percent Variance':'1','Principal Component n':'5'}
    with open(os.path.expanduser('~')+'\\Documents\\Autosort.ini','w') as configfile:
        config.write(configfile)
    