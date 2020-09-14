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

config_ver=5

def do_the_config(path=''):
    if os.name=='posix':
        path=os.path.expanduser('~')+'/Autosort/Autosort_config.ini'
        if not os.path.isfile(path):
            linux_config(path)
        return read_config(path)
    elif os.name=='nt':
        if path=='':
            path=os.path.expanduser('~')+'\\Documents\\Autosort/Autosort_config.ini'
        if not os.path.isfile(path):
            default_config(path)
            print('Default configuration file has been created. You can find it in '+path)
        else: return read_config(path)

def rec_info(fullpath):
    config=configparser.ConfigParser()
    config.read(fullpath)
    return config['METADATA']['Recording Type']

def default_config(path):
    if not os.path.isdir(os.path.split(path)[0]):
        os.mkdir(os.path.split(path)[0])
    config=configparser.ConfigParser()
    config['Run Settings']={'Resort Limit':'3','Minimum Licks':'1000','Cores Used':'8',
                            'Weekday Run':'2','Weekend Run':'8','Run Type':'Auto','Manual Run':'2'}
    config['Paths']={'Pl2 To-Run Path':r'R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted','Running Path':r'R:\Dannymarsh Sorting Emporium\running_files',
                     'Results Path':r'R:\Dannymarsh Sorting Emporium\Results','Completed Pl2 Path':r'R:\Dannymarsh Sorting Emporium\completed_pl2',
                     'Use Paths':'1','Else Path':''}
    config['Clustering']={'Max Clusters':'7','Max Iterations':'1000','Convergence Criterion':'.0001','Random Restarts':'10','L-ratio Cutoff':'.1'}
    config['Signal']={'Disconnect Voltage':'1500','Max Breach Rate':'.2','Max Breach Count':'10','Max Breach Avg.':'20','Intra-Cluster Cutoff':'3'}
    config['Filtering']={'Low Cutoff':'300', 'High Cutoff':'6000'}
    config['Spike']={'Pre-time':'.2','Post-time':'.6','Sampling Rate':'40000'}
    config['Std Dev']={'Spike Detection':'2.0','Artifact Removal':'10.0'}
    config['PCA']={'Variance Explained':'.95','Use Percent Variance':'1','Principal Component n':'5'}
    config['Post Process']={'reanalyze':'0','simple gmm':'1','image size':'70','temporary dir':os.path.expanduser('~')+'\\tmp_python'}
    config['Version']={'config version':str(config_ver)}
    with open(path,'w') as configfile:
        config.write(configfile)

def linux_config(path):
    config=configparser.ConfigParser()
    config['Run Settings']={'Resort Limit':'3','Cores Used':'8','N-Files':'1'}
    config['Paths']={'h5 To-Run Path':'/data/home/dmarshal/Autosort/h5_for_run','Running Path':'/data/home/dmarshal/Autosort/running_files',
                     'Results Path':'/data/home/dmarshal/Autosort/Results'}
    config['Clustering']={'Max Clusters':'7','Max Iterations':'1000','Convergence Criterion':'.0001','Random Restarts':'10','L-ratio Cutoff':'.1'}
    config['Signal']={'Disconnect Voltage':'1500','Max Breach Rate':'.2','Max Breach Count':'10','Max Breach Avg.':'20','Intra-Cluster Cutoff':'3'}
    config['Filtering']={'Low Cutoff':'300', 'High Cutoff':'6000'}
    config['Spike']={'Pre-time':'.2','Post-time':'.6','Sampling Rate':'40000'}
    config['Std Dev']={'Spike Detection':'2.0','Artifact Removal':'10.0'}
    config['PCA']={'Variance Explained':'.95','Use Percent Variance':'1','Principal Component n':'5'}
    config['Version']={'config version':str(config_ver)}
    with open(path,'w') as configfile:
        config.write(configfile)
        
def read_config(path):
    config=configparser.ConfigParser()
    params={}
    config.read(path)
    for key,value in config._sections.items():
        params.update(value)
    if config_ver!=int(params['config version']):
        os.rename(path,os.path.splitext(path)[0]+str(params['config version'])+'.txt')
        default_config(path)
        print('Config version updated, config file reset to default, your original config file has been renamed. Find the new config file here: '+path)
    else:
        return params
    
    