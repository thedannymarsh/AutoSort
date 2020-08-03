import numpy as np
from scipy.signal import butter
from scipy.signal import filtfilt
from scipy.interpolate import interp1d
from sklearn.mixture import GaussianMixture
import pylab as plt
from sklearn.decomposition import PCA
import os
import subprocess
import pandas as pd
import shutil

def get_filtered_electrode(el, freq = [300, 6000.0], sampling_rate = 40000.0):   #############################    MODIFIED    #####################################
    m, n = butter(2, [2.0*freq[0]/sampling_rate, 2.0*freq[1]/sampling_rate], btype = 'bandpass') 
    filt_el = filtfilt(m, n, el)
    return filt_el

def extract_waveforms(filt_el, spike_snapshot = [0.2, 0.6], sampling_rate = 40000.0, STD = 2.0, cutoff_std=10.0): 
    m = np.mean(filt_el)
    th = np.std(filt_el) * STD ################### was 5.0*np.median(np.abs(filt_el)/0.6745)
    pos = np.where(filt_el <= m-th)[0]
    changes = []
    for i in range(len(pos)-1):
        if pos[i+1] - pos[i] > 1:
            changes.append(i+1)

    # slices = np.zeros((len(changes)-1,150))
    slices = []
    spike_times = []
    for i in range(len(changes) - 1):
        minimum = np.where(filt_el[pos[changes[i]:changes[i+1]]] == np.min(filt_el[pos[changes[i]:changes[i+1]]]))[0]
        #print minimum, len(slices), len(changes), len(filt_el)
        # try slicing out the putative waveform, only do this if there are 10ms of data points (waveform is not too close to the start or end of the recording)
        if pos[minimum[0]+changes[i]] - int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0)) > 0 and pos[minimum[0]+changes[i]] + int((spike_snapshot[1] + 0.1)*(sampling_rate/1000.0)) < len(filt_el):
            tempslice=filt_el[pos[minimum[0]+changes[i]] - int((spike_snapshot[0] + 0.1)*(sampling_rate/1000.0)) : pos[minimum[0]+changes[i]] + int((spike_snapshot[1] + 0.1)*(sampling_rate/1000.0))]
            if ~np.any(np.absolute(tempslice) > (th*cutoff_std)/STD):
                slices.append(tempslice)
                spike_times.append(pos[minimum[0]+changes[i]])

    return np.array(slices), spike_times

def dejitter(slices, spike_times, spike_snapshot = [0.2, 0.6], sampling_rate = 40000.0):
    x = np.arange(0,len(slices[0]),1)
    xnew = np.arange(0,len(slices[0])-1,0.1)

    # Calculate the number of samples to be sliced out around each spike's minimum
    before = int((sampling_rate/1000.0)*(spike_snapshot[0]))
    after = int((sampling_rate/1000.0)*(spike_snapshot[1]))
    
    #slices_dejittered = np.zeros((len(slices)-1,300))
    slices_dejittered = []
    slices_broken = []
    spike_times_dejittered = []
    for i in range(len(slices)):
        f = interp1d(x, slices[i])
        # 10-fold interpolated spike
        ynew = f(xnew)
        minimum = np.where(ynew == np.min(ynew))[0][0]
        # Only accept spikes if the interpolated minimum has shifted by less than 1/10th of a ms (4 samples for a 40kHz recording, 40 samples after interpolation)
        # If minimum hasn't shifted at all, then minimum - 5ms should be equal to zero (because we sliced out 5 ms before the minimum in extract_waveforms())
        # We use this property in the if statement below
        if np.abs(minimum - int((spike_snapshot[0] + 0.1)*(sampling_rate/100.0))) < int(10.0*(sampling_rate/10000.0)):
            slices_dejittered.append(ynew[minimum - before*10 : minimum + after*10])
            spike_times_dejittered.append(spike_times[i])
    return np.array(slices_dejittered), np.array(spike_times_dejittered)

def scale_waveforms(slices_dejittered):
    energy = np.sqrt(np.sum(slices_dejittered**2, axis = 1))/len(slices_dejittered[0])
    scaled_slices = np.zeros((len(slices_dejittered),len(slices_dejittered[0])))
    for i in range(len(slices_dejittered)):
        scaled_slices[i] = slices_dejittered[i]/energy[i]

    return scaled_slices, energy

def implement_pca(scaled_slices):
    pca = PCA()
    pca_slices = pca.fit_transform(scaled_slices)    
    return pca_slices, pca.explained_variance_ratio_

def clusterGMM(data, n_clusters, n_iter, restarts, threshold):
    

    g = []
    bayesian = []

    for i in range(restarts):
        g.append(GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = threshold, random_state = i, max_iter = n_iter))
        g[-1].fit(data)
        if g[-1].converged_:
            bayesian.append(g[-1].bic(data))
        else:
            del g[-1]

    #print len(akaike)
    bayesian = np.array(bayesian)
    best_fit = np.where(bayesian == np.min(bayesian))[0][0]
    
    predictions = g[best_fit].predict(data)

    return g[best_fit], predictions, np.min(bayesian)

def isoinfo(data,predictions,isodir='temporary_iso_info',Lrat_cutoff=.1):   
    #runs isolation information processing on feature data, and returns cluster isolation data in the form of a pandas dataframe
    if os.path.isdir(isodir):
        shutil.rmtree(isodir)
    try: os.mkdir(isodir) #make the temp directory
    except: pass
    olddir=os.getcwd()
    os.chdir(isodir)
    #feature data packages the predictions and features in the format required by the isorat and isoi executables
    featuredata=pd.DataFrame(np.concatenate((np.reshape(predictions+1,(len(predictions),1)), data),axis=1),columns=['Cluster','Energy','Amplitude']+['PC'+str(n) for n in range(1,np.shape(data)[1]-1)])
    featuredata=featuredata.astype({'Cluster':int})
    featuredata.to_csv(isodir+'/featuredata.txt',header=True, index=False, sep='\t') 
    subprocess.run([os.path.split(__file__)[0]+'/bin/isorat.exe',"featuredata.txt",'isorat_output.txt'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #run isorat (produces isod and l ratio)
    isorat_df=pd.read_csv('isorat_output.txt',sep=' ',names=['IsoD','L-Ratio'])
    if (isorat_df['L-Ratio']>Lrat_cutoff).all() == False:
        subprocess.run([os.path.split(__file__)[0]+'/bin/isoi.exe',"featuredata.txt",'isoi_output.txt'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) #run isoi
        iso=pd.concat((pd.read_csv('isoi_output.txt',sep=' ',names=['IsoIBG','IsoINN','NNClust']),isorat_df),axis=1)
        iso=iso.add([0,0,-1,0,0])
    else:
        isorat_df.insert(0,'NNClust','nan')
        isorat_df.insert(0,'IsoINN','nan')
        isorat_df.insert(0,'IsoIBG','nan')
        iso=isorat_df
    iso.insert(0,'Cluster',iso.index) #reads and packages the data in a usable dataframe
    os.chdir(olddir)
    shutil.rmtree(isodir)
    return iso