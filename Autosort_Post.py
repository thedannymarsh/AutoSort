import os
import tables
import numpy as np
import easygui
import ast
import pylab as plt
from sklearn.mixture import GaussianMixture
import pypl2.Pl2_waveforms_datashader
import pypl2.Clustering as clust
from pypl2 import config_handler
import json
import shutil
import sys
import pandas as pd
from datetime import date

#Get parameters
params=config_handler.do_the_config()
reanalyze=int(params['reanalyze'])
simple_GMM=int(params['simple gmm'])
temp_dir=params['temporary dir']
image_size=int(params['image size'])

#If the image directory does not exit, create it
if os.path.isdir(temp_dir):
    shutil.rmtree(temp_dir)
os.mkdir(temp_dir)
figname=temp_dir+'/tempfig.png'
dsdir=temp_dir+'/ds'

# Check for user name and ensure user has a folder for data to go into
os.chdir("R:\Autobots Roll Out") # Go to data folder
folderlist = os.listdir('./' ) # Check for all folders in data folder

# Create list of all folders (users)
userlist=[]
for item in folderlist:
    if len(os.path.splitext(item)[1])==0:
        userlist.append(item)


# Ask current user for their user name
UserName = easygui.choicebox(msg= 'Please select a user name, or click cancel to create a new user',choices=userlist)
if UserName is None:
    UserName = easygui.enterbox(msg = 'Please enter a new username. This is used to identify where your data is stored.')
    if UserName is None or UserName in userlist:
        easygui.msgbox(msg=str(UserName)+" is an invalid choice.")
        sys.exit()

# If folder does not have a folder, create it
if os.path.isdir("R:\\Autobots Roll Out\\%s" %UserName) == False:
    os.mkdir("R:\\Autobots Roll Out\\%s" %UserName)
    os.mkdir("R:\\Autobots Roll Out\\%s\\JSON_Files" %UserName)
    os.mkdir("R:\\Autobots Roll Out\\%s\\NewNexFiles" %UserName)
if not os.path.isdir("R:\\Autobots Roll Out\\"+UserName+'/Info_Files'): 
    os.mkdir("R:\\Autobots Roll Out\\"+UserName+'/Info_Files')

# Determine which files have already been analyzed
os.chdir("R:\\Autobots Roll Out\\%s\\JSON_Files" %UserName) # Change directory to folder containing all json files
Analyzedlist = os.listdir('./' ) # Check for all folders in data folder

# Get directory where the hdf5 file sits, and change to that directory
check = easygui.msgbox(msg = 'Hello %s, please select the folder containing the file(s) you want to check.' %UserName)

filedir = easygui.diropenbox()
os.chdir(filedir)

FilesAnalyzed = ''
Redo = True # If no json files exist, redo everything
DelOld = True # If files have been worked on but not saved, delete old unitsJosh
if len(Analyzedlist) >0: # If there are any json files in the analyzed folder
    for n in range(len(Analyzedlist)):
        if n == 0: # If the first file
            FilesAnalyzed = str(Analyzedlist[n][:-5])
        else:6
if reanalyze==1:
    DelOld = easygui.ynbox(msg = 'Do you want to delete old cells in the files you have already done? The files are: \n\n' + FilesAnalyzed) # Ask the user if they want to delete old cells
    if DelOld == False: # If they do not...
        Redo = easygui.ynbox(msg = 'Do you want to add new cells to the files you have already done?') # Ask if they want to add any cells
elif reanalyze==0:
    DelOld=False
    Redo=False

# Look for the hdf5 file in the directory
file_list = os.listdir('./')
hdf5_name = ''
first = True # Indicating first file

for files in file_list:
    file_iso=pd.DataFrame(columns=['IsoRating','File','Channel','Solution','Cluster','wf count','ISIs (%)','L-Ratio','Post-Process Date','Recording Type'])
    skip = False # Set skipping already done files to false
    if files[-2:] == 'h5': # If a specific file is an .h5 file
        hdf5_name = files # Make that the current file
        if Redo == False:  # If user did not want to redo the already done files
            for done in Analyzedlist: # For all analyzed files
                if hdf5_name[:-3] == done[:-5]: # If current file matched the name
                    skip = True # Set skip to true
        if skip == True: # If skip is true
            continue # Skip current file
            
        dir_name = "%s\\%s" % (filedir, hdf5_name[:-3]) # Set directory to current file
        
        if first == True: # If on the first file
            check = easygui.msgbox(msg = 'The first file is %s. ' % hdf5_name) # Simply tell user file name
            first = False # Set first to False since it will no longer be the first file
        else: # If not the first file
            check = easygui.ynbox(msg = 'Do you want to continue onto the next file? The file is %s?' % hdf5_name) # Ask user if they want to keep going
        if check == False: # If they do not want to continue
            break # Exit program

        # Open the hdf5 file
        hf5 = tables.open_file(hdf5_name, 'r+')
        # Delete the raw nodes, if they exist in the hdf5 file, to cut down on file size
        if DelOld == True:
            try:
                hf5.remove_node('/sorted_units', recursive = 1) 
            except:
                pass
        
        try:
            elNum = len(hf5.list_nodes("/SPKC"))
            hf5.remove_node('/SPKC', recursive = 1)
            # And if successful, close the currently open hdf5 file and ptrepack the file
            hf5.close()
            print("SPKC recordings removed")
            #os.system("ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc " + hdf5_name + " " + hdf5_name[:-3] + "_repacked.h5")
            # Delete the old (SPKC and big) hdf5 file
            #os.system("rm " + hdf5_name)
            # And open the new, repacked file
            hf5 = tables.open_file(hdf5_name[:-3] + ".h5", 'r+')
            print("File repacked")
        except:
            print("SPKC recordings have already been removed, so moving on ..")
            elNum = 16
        
        # Make the sorted_units group in the hdf5 file if it doesn't already exist
        try:
            hf5.create_group('/', 'sorted_units')
        except:
            pass
        
        # Define a unit_descriptor class to be used to add things (anything!) about the sorted units to a pytables table
        class unit_descriptor(tables.IsDescription):
            electrode_number = tables.Int32Col()
            single_unit = tables.Int32Col()
            regular_spiking = tables.Int32Col()
            fast_spiking = tables.Int32Col()
        
        # Make a table under /sorted_units describing the sorted units. If unit_descriptor already exists, just open it up in the variable table
        try:
            table = hf5.create_table('/', 'unit_descriptor', description = unit_descriptor)
        except:
            table = hf5.root.unit_descriptor
        
        # Run an infinite loop as long as the user wants to pick clusters from the electrodes	
        while True:
            try:
                print('Current Units for: {}'.format(hdf5_name))
                print(file_iso[['Channel','Solution','Cluster','wf count','ISIs (%)','L-Ratio']])   
                # Get electrode number from user
                electrode_num = easygui.multenterbox(msg = 'For file %s, which electrode (1-%s) do you want to choose? \nHit cancel to move to the next file' % (hdf5_name[:-3], str(elNum)), fields = ['Electrode #'])
                # Break if wrong input/cancel command was given
                try:
                    electrode_num = int(electrode_num[0])-1
                except:
                    break
            	
                # Get the number of clusters in the chosen solution
                num_clusters = easygui.multenterbox(msg = 'Which solution do you want to choose for electrode %i?' % (electrode_num+1), fields = ['Number of clusters in the solution'])
                num_clusters = int(num_clusters[0])
 
                # Get cluster choices from the chosen solution
                clusters = easygui.multchoicebox(msg = 'For file %s, electrode %i, in the %i cluster batch, Which clusters do you want to choose?' % (hdf5_name[:-3], electrode_num+1, num_clusters), choices = tuple([str(i) for i in range(num_clusters)]))
    	
                # Check if the user wants to merge clusters if more than 1 cluster was chosen. Else ask if the user wants to split/re-cluster the chosen cluster
                merge = False
                re_cluster = False
                if len(clusters) > 1:
                    merge = str(easygui.ynbox(msg = 'Do you want to merge these clusters into one unit?'))
                    merge = ast.literal_eval(merge)
                else:
                    re_cluster = str(easygui.ynbox(msg = 'Do you want to split this cluster?'))
                    re_cluster = ast.literal_eval(re_cluster)
                        
                # Load data from the chosen electrode and solution
                spike_waveforms = np.load('./'+ hdf5_name[:-3] +'/spike_waveforms/electrode %i/spike_waveforms.npy' % (electrode_num+1))
                spike_times = np.load('./'+ hdf5_name[:-3] +'/spike_times/electrode %i/spike_times.npy' % (electrode_num+1))
                energy = np.load('./'+ hdf5_name[:-3] +'/spike_waveforms/electrode %i/energy.npy' % (electrode_num+1))
                amplitudes = np.load('./'+ hdf5_name[:-3] +'/spike_waveforms/electrode %i/spike_amplitudes.npy' % (electrode_num+1))
                predictions = np.load('./'+ hdf5_name[:-3] +'/clustering_results/electrode %i/clusters%i/predictions.npy' % ((electrode_num+1), num_clusters))
   
            
                # Scale the dejittered slices by the energy of the waveforms
                scaled_slices, energy = clust.scale_waveforms(spike_waveforms)
                
                # Run PCA on the scaled waveforms
                pca_slices, explained_variance_ratio = clust.implement_pca(scaled_slices)
                
                #get cumulative variance explained
                cumulvar=np.cumsum(explained_variance_ratio)
                pvar=float(params['variance explained'])
                usepvar=int(params['use percent variance'])
                userpc=int(params['principal component n'])
                if usepvar==1:n_pc=np.where(cumulvar>pvar)[0][0]+1
                else: n_pc=userpc
                
            	# If the user asked to split/re-cluster, ask them for the clustering parameters and perform clustering
                split_predictions = []
                chosen_split = 0
                if re_cluster: 
            		# Get clustering parameters from user
                    if simple_GMM==0:
                        clustering_params = easygui.multenterbox(msg = 'Fill in the parameters for re-clustering (using a GMM)', fields = ['Number of clusters', 'Maximum number of iterations (1000 is more than enough)', 'Convergence criterion (usually 0.0001)', 'Number of random restarts for GMM (10 is more than enough)'])
                        n_clusters = int(clustering_params[0])
                        n_iter = int(clustering_params[1])
                        thresh = float(clustering_params[2])
                        n_restarts = int(clustering_params[3])          
                    elif simple_GMM==1:
                        n_clusters = int(easygui.multenterbox(msg = 'Enter a number of target clusters (using a GMM) \n\nNote: You have selected easy GMM \nMaximum Iterations=1000 \nConvergence Criterion=.0001 \nNumber of random restarts=10', fields = ['Number of clusters'])[0])
                        n_iter = 1000
                        thresh = .0001
                        n_restarts = 10
            		# Make data array to be put through the GMM - 5 components: 3 PCs, scaled energy, amplitude
                    this_cluster = np.where(predictions == int(clusters[0]))[0]
                    data = np.zeros((len(this_cluster), n_pc + 2))	
                    data[:,2:] = pca_slices[this_cluster,:n_pc]
                    data[:,0] = energy[this_cluster]/np.max(energy[this_cluster])
                    data[:,1] = np.abs(amplitudes[this_cluster])/np.max(np.abs(amplitudes[this_cluster]))
            
            		# Cluster the data
                    g = GaussianMixture(n_components = n_clusters, covariance_type = 'full', tol = thresh, max_iter = n_iter, n_init = n_restarts)
                    g.fit(data)
            	
            		# Show the cluster plots if the solution converged
                    if g.converged_:
                        split_predictions = g.predict(data)
                        x = np.arange(len(spike_waveforms[0])/10) + 1
                        for cluster in range(n_clusters):
                            split_points = np.where(split_predictions == cluster)[0]				
            				# plt.figure(cluster)
                            slices_dejittered = spike_waveforms[this_cluster, :]		# Waveforms and times from the chosen cluster
                            times_dejittered = spike_times[this_cluster]
                            times_dejittered = times_dejittered[split_points]		# Waveforms and times from the chosen split of the chosen cluster
                            ISIs = np.ediff1d(np.sort(times_dejittered))/40.0       # Get number of points between waveforms and divide by frequency per ms (40)
                            violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                            violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                            fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(slices_dejittered[split_points, :], x,dir_name=dsdir)
            				# plt.plot(x-15, slices_dejittered[split_points, :].T, linewidth = 0.01, color = 'red')
                            ax.set_xlabel('Sample (40 points per ms)')
                            ax.set_ylabel('Voltage (microvolts)')
                            ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                    else:
                        print("Solution did not converge - try again with higher number of iterations or lower convergence criterion")
                        continue
            
                    plt.show()
            		# Ask the user for the split clusters they want to choose
                    chosen_split = easygui.multchoicebox(msg = 'Which split cluster(s) do you want to choose? Hit cancel to return to electrode selection.', choices = tuple([str(i) for i in range(n_clusters)]))
                    try:
                        len(chosen_split)
                        those_cluster = np.where(predictions != int(clusters[0]))[0]
                        thosedata = np.zeros((len(those_cluster), n_pc + 2))	
                        thosedata[:,2:] = pca_slices[those_cluster,:n_pc]
                        thosedata[:,0] = energy[those_cluster]/np.max(energy[those_cluster])
                        thosedata[:,1] = np.abs(amplitudes[those_cluster])/np.max(np.abs(amplitudes[those_cluster]))
                        alldata=np.concatenate((thosedata,data),axis=0)
                        all_predictions=np.concatenate((np.zeros(np.shape(thosedata)[0])-1,split_predictions))
                        Lrats=clust.get_Lratios(alldata,all_predictions)
                        ISIList=[]
                        for choice in chosen_split:
                            cluster=int(choice)
                            split_points = np.where(split_predictions == cluster)[0]				
            				# plt.figure(cluster)
                            slices_dejittered = spike_waveforms[this_cluster, :]		# Waveforms and times from the chosen cluster
                            times_dejittered = spike_times[this_cluster]
                            times_dejittered = times_dejittered[split_points]		# Waveforms and times from the chosen split of the chosen cluster
                            ISIs = np.ediff1d(np.sort(times_dejittered))/40.0       # Get number of points between waveforms and divide by frequency per ms (40)
                            violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                            ISIList.append(round(violations1,1))
                            violations2 = 100.0*float(np.sum(ISIs < 2.0)/split_points.shape[0])
                            fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(slices_dejittered[split_points, :], x,dir_name=dsdir)
            				# plt.plot(x-15, slices_dejittered[split_points, :].T, linewidth = 0.01, color = 'red')
                            ax.set_xlabel('Sample (40 points per ms)')
                            ax.set_ylabel('Voltage (microvolts)')
                            ax.set_title("Split Cluster{:d}, 2ms ISI violations={:.1f} percent".format(cluster, violations2) + "\n" + "1ms ISI violations={:.1f}%, Number of waveforms={:d}".format(violations1, split_points.shape[0]))
                            fig.savefig(figname,dpi=image_size)
                            clcheck=easygui.ynbox(msg='Please verify that this cluster is correct.\nL-Ratio: {}'.format(round(Lrats[int(choice)],3)),image=figname)
                            if clcheck: pass
                            else: 
                                easygui.msgbox('You indicated that a split cluster was not correct. Results have not been saved.')
                                continue
                    except:
                        continue
            
            	# Get list of existing nodes/groups under /sorted_units
                node_list = hf5.list_nodes('/sorted_units')
            
            	# If node_list is empty, start naming units from 001
                unit_name = ''
                max_unit = 1
                unit_chan = ['%02d' % (electrode_num +1)]
                if node_list == []:		
                    unit_name = 'unit%03d' % int(max_unit)
            	# Else name the new unit by incrementing the last unit by 1 
                else:
                    unit_numbers = []
                    for node in node_list:
                        unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
                        unit_numbers[-1] = int(unit_numbers[-1])
                    unit_numbers = np.array(unit_numbers)
                    max_unit = np.max(unit_numbers)
                    unit_name = 'unit%03d' % int(max_unit + 1)
    
            	# Get a new unit_descriptor table row for this new unit
                unit_description = table.row	
            
            	# If the user re-clustered/split clusters, add the chosen clusters in split_clusters
                if re_cluster:
                    p=0
                    for cluster in range(len(chosen_split)):
                        p = p+1
                        if p>1:
                            unit_numbers = []
                            node_list = hf5.list_nodes('/sorted_units')
                            for node in node_list:
                                unit_numbers.append(node._v_pathname.split('/')[-1][-3:])
                                unit_numbers[-1] = int(unit_numbers[-1])
                            unit_numbers = np.array(unit_numbers)
                            max_unit = np.max(unit_numbers)
                            unit_name = 'unit%03d' % int(max_unit + 1)
                        hf5.create_group('/sorted_units', unit_name)
                        unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]	# Waveforms of originally chosen cluster
                        unit_waveforms = unit_waveforms[np.where(split_predictions == int(chosen_split[int(cluster)]))[0], :]	# Subsetting this set of waveforms to include only the chosen split
                        unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]			# Do the same thing for the spike times
                        unit_times = unit_times[np.where(split_predictions == int(chosen_split[int(cluster)]))[0]]
                        waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                        times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                        channel = hf5.create_array('/sorted_units/%s' % unit_name, 'channel', unit_chan)
                        unit_description['electrode_number'] = electrode_num
                        # single_unit = str(easygui.ynbox(msg = 'Are you mostly-SURE that %s is a beautiful single unit?' % chosen_split[cluster]))
                        unit_description['single_unit'] = int(1)
                		# If the user says that this is a single unit, ask them whether its regular or fast spiking
                        unit_description['regular_spiking'] = 1
                        unit_description['fast_spiking'] = 0
                        # if int(ast.literal_eval(single_unit)):
                        #     unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                        #     unit_description[unit_type[0]] = 1		
                        unit_description.append()
                        table.flush()
                        hf5.flush()
                        choice=int(chosen_split[cluster])
                        split_points = np.where(split_predictions == choice)[0]
                        violations1 = 100.0*float(np.sum(ISIs < 1.0)/split_points.shape[0])
                        this_iso={
                            'IsoRating':'TBD',
                            'File':os.path.splitext(hdf5_name)[0],
                            'Channel':electrode_num+1,
                            'Solution':num_clusters,
                            'Cluster':'s',
                            'wf count':len(np.where(split_predictions == int(chosen_split[int(cluster)]))[0]),
                            'ISIs (%)': ISIList[cluster],
                            'L-Ratio':round(Lrats[int(chosen_split[cluster])],3),
                            }
                        file_iso=file_iso.append(this_iso,ignore_index=True)

            
            	# If only 1 cluster was chosen (and it wasn't split), add that as a new unit in /sorted_units. Ask if the isolated unit is an almost-SURE single unit
                elif len(clusters) == 1:
                    unit_waveforms = spike_waveforms[np.where(predictions == int(clusters[0]))[0], :]
                    x = np.arange(len(unit_waveforms[0])/10) + 1
                    fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(unit_waveforms, x,dir_name=dsdir)
                    ax.set_xlabel('Sample (40 points per ms)')
                    ax.set_ylabel('Voltage (microvolts)')
                    ax.set_title("Channel: "+str(electrode_num+1)+", Solution: "+str(num_clusters)+", Cluster: "+str(clusters[0]))
                    fig.savefig(figname,dpi=image_size)
                    iso=pd.read_csv(hdf5_name[:-3]+'/clustering_results/electrode {}/clusters{}/isoinfo.csv'.format(electrode_num+1,num_clusters))
                    unit_verify = easygui.ynbox(msg = "Please verify that this is the correct unit.\nL-Ratio: {}".format(iso['L-Ratio'][int(clusters[0])]),image=figname)
                    if unit_verify:
                        file_iso=file_iso.append(iso.loc[int(clusters[0])],ignore_index=True)
                        hf5.create_group('/sorted_units', unit_name)
                        unit_times = spike_times[np.where(predictions == int(clusters[0]))[0]]
                        waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                        times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                        channel = hf5.create_array('/sorted_units/%s' % unit_name, 'channel', unit_chan)
                        unit_description['electrode_number'] = electrode_num
                        unit_description['single_unit'] = int(1)
                		# If the user says that this is a single unit, ask them whether its regular or fast spiking
                        unit_description['regular_spiking'] = 1
                        unit_description['fast_spiking'] = 0
                        # if int(ast.literal_eval(single_unit)):
                        #     unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                        #     unit_description[unit_type[0]] = 1
                        unit_description.append()
                        table.flush()
                        hf5.flush()
                    else: 
                        easygui.msgbox('You indicated that this cluster was incorrect. Results for this channel have not been saved.')
                        del unit_waveforms
                        continue
            
                else:
            		# If the chosen units are going to be merged, merge them
                    if merge:
                        unit_waveforms = None
                        unit_times = None
                        for cluster in clusters:
                            if unit_waveforms is None:
                                unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]			
                                unit_times = spike_times[np.where(predictions == int(cluster))[0]]
                            else:
                                unit_waveforms = np.concatenate((unit_waveforms, spike_waveforms[np.where(predictions == int(cluster))[0], :]))
                                unit_times = np.concatenate((unit_times, spike_times[np.where(predictions == int(cluster))[0]]))
               
                        data = np.zeros((len(pca_slices), n_pc + 2))
                        data[:,2:] = pca_slices[:,:n_pc]
                        data[:,0] = energy[:]/np.max(energy)
                        data[:,1] = np.abs(amplitudes)/np.max(np.abs(amplitudes))
                        merge_predictions=np.array([int(clusters[0]) if str(x) in clusters else x for x in predictions])
                        Lrats=clust.get_Lratios(data,merge_predictions)
                        
            			# Show the merged cluster to the user, and ask if they still want to merge
                        x = np.arange(len(unit_waveforms[0])/10) + 1
                        fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(unit_waveforms, x,dir_name=dsdir)
            			# plt.plot(x - 15, unit_waveforms[:, ::10].T, linewidth = 0.01, color = 'red')
                        ax.set_xlabel('Sample (40 points per ms)')
                        ax.set_ylabel('Voltage (microvolts)')
                        fig.savefig(figname,dpi=image_size)
                        
            			# Warn the user about the frequency of ISI violations in the merged unit
                        ISIs = np.ediff1d(np.sort(unit_times))/40.0       # Get number of points between waveforms and divide by frequency per ms (40)
                        violations1 = 100.0*float(np.sum(ISIs < 1.0)/len(unit_times))
                        violations2 = 100.0*float(np.sum(ISIs < 2.0)/len(unit_times))
                        proceed = easygui.ynbox(image=figname,msg = 'The merged cluster has %.1f percent (<2ms) and %.1f percent (<1ms) ISI violations out of %i total waveforms. Do you still want to merge these clusters into one unit?' % (violations2, violations1, len(unit_times))+'\nL-ratio of merged clusters:{}'.format(round(Lrats[int(clusters[0])],3)))
                        
            			# Create unit if the user agrees to proceed, else abort and go back to start of the loop 
                        if proceed:	
                            this_iso={
                                'IsoRating':'TBD',
                                'File':os.path.splitext(hdf5_name)[0],
                                'Channel':electrode_num+1,
                                'Solution':num_clusters,
                                'Cluster':'m',
                                'wf count':len(np.where(merge_predictions==int(clusters[0]))[0]),
                                'ISIs (%)':round(violations1,1),
                                'L-Ratio':round(Lrats[int(clusters[0])],3),
                                }
                            file_iso=file_iso.append(this_iso,ignore_index=True)
                            hf5.create_group('/sorted_units', unit_name)
                            waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                            times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                            channel = hf5.create_array('/sorted_units/%s' % unit_name, 'channel', unit_chan)
                            unit_description['electrode_number'] = electrode_num
                            unit_description['single_unit'] = int(1)
            				# If the user says that this is a single unit, ask them whether its regular or fast spiking
                            unit_description['regular_spiking'] = 1
                            unit_description['fast_spiking'] = 0
                            # if int(ast.literal_eval(single_unit)):
                            #     unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                            #     unit_description[unit_type[0]] = 1
                            unit_description.append()
                            table.flush()
                            hf5.flush()
                        else:
                            easygui.msgbox('You indicated that the merged cluster was not isolated, results for this channel have not been saved.')
                            continue
            
            		# Otherwise include each cluster as a separate unit
                    else:
                        bad_cluster=False
                        for cluster in clusters:
                            unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
                            x = np.arange(len(unit_waveforms[0])/10) + 1
                            fig, ax = pypl2.Pl2_waveforms_datashader.waveforms_datashader(unit_waveforms, x,dir_name=dsdir)
                            ax.set_xlabel('Sample (40 points per ms)')
                            ax.set_ylabel('Voltage (microvolts)')
                            ax.set_title("Channel: "+str(electrode_num+1)+", Solution: "+str(num_clusters)+", Cluster: "+str(cluster))
                            fig.savefig(figname,dpi=image_size)
                            iso=pd.read_csv(hdf5_name[:-3]+'/clustering_results/electrode {}/clusters{}/isoinfo.csv'.format(electrode_num+1,num_clusters))
                            unit_verify = str(easygui.ynbox(msg = "Please verify that this is the correct unit.\nL-Ratio: {}".format(iso['L-Ratio'][int(cluster)]),image=figname))
                            if ast.literal_eval(unit_verify): pass
                            else: 
                                bad_cluster=True
                                break
                        del unit_waveforms
                        if bad_cluster==False:
                            for cluster in clusters:
                                file_iso=file_iso.append(iso.loc[int(cluster)],ignore_index=True)
                                hf5.create_group('/sorted_units', unit_name)
                                unit_waveforms = spike_waveforms[np.where(predictions == int(cluster))[0], :]
                                unit_times = spike_times[np.where(predictions == int(cluster))[0]]
                                waveforms = hf5.create_array('/sorted_units/%s' % unit_name, 'waveforms', unit_waveforms)
                                times = hf5.create_array('/sorted_units/%s' % unit_name, 'times', unit_times)
                                channel = hf5.create_array('/sorted_units/%s' % unit_name, 'channel', unit_chan)
                                unit_description['electrode_number'] = electrode_num
                                unit_description['single_unit'] = int(1)
                				# If the user says that this is a single unit, ask them whether its regular or fast spiking
                                unit_description['regular_spiking'] = 1
                                unit_description['fast_spiking'] = 0
                                # if int(ast.literal_eval(single_unit)):
                                #     unit_type = easygui.multchoicebox(msg = 'What type of unit is this (Regular spiking = Pyramidal cells, Fast spiking = PV+ interneurons)?', choices = ('regular_spiking', 'fast_spiking'))
                                #     unit_description[unit_type[0]] = 1
                                unit_description.append()
                                table.flush()
                                hf5.flush()				
                
                				# Finally increment max_unit and create a new unit name
                                max_unit += 1
                                unit_name = 'unit%03d' % int(max_unit + 1)
                
                				# Get a new unit_descriptor table row for this new unit
                                unit_description = table.row
                        else: 
                            easygui.msgbox("You indicated that one of the clusters was incorrect. Results for this channel have not been saved.")
                            continue
            except Exception as e:
                easygui.msgbox("An error occured, you will be returned to electrode selection\n\nError message:"+str(e))
                
        #################   Initialize Data Collection For JSON File   ########################
        
        filename = ("R:\\Autobots Roll Out\\%s\\NewNexFiles\\%s" % (UserName, hdf5_name[:-3]))
        data = {
            "FileName": filename,
            "Neurons": [],
            "Events":[],
            "Waveforms":[],
    	}
    
    	#################   Get Event Names   ########################
        evtNames = [] # For holding event names
        for n in hf5.list_nodes("/events"): # Look in the events group for all events
            evtNames.append(str(n))
        
        tempName0 = [0]*len(evtNames) # For splitting events names
        tempName1 = [0]*len(evtNames) # For splitting events names
        for n in range(len(evtNames)): 
            tempName0[n] = str.split(evtNames[n], '/') # Split event names whereever "/" exists
            evtNames[n] = str.split(tempName0[n][2], '(')[0][:-1] #get event names #change to work with spaces in tastant names
    
            #################   Get Event Timestamps   ########################
    
            event=getattr(hf5.root.events,evtNames[n])[:]# Get timestamps for each event
            data["Events"].append({ # Append data to the data variable in JSON format
                    evtNames[n]: event
                    })

    	#################   Get Spike Names and Export to JSON File   ########################
    
        spkNames = [] # For holding spike names
        for n in hf5.list_nodes("/sorted_units"): # Look in sorted_units group for all sorted neurons
            spkNames.append(str(n))
    
        tempName0 = [0]*len(spkNames) # For splitting spike names
        tempName1 = [0]*len(spkNames) # For splitting spike names
        spkTimes = [] # For holding spike timestamps
        spkWaves = [] # For holding spike waveforms
        olchan = [b'-01']
        char = 97
        for n in range(len(spkNames)): 
            tempName0[n] = str.split(spkNames[n], '/') # Split event names whereever "/" exists
            tempName1[n] = str.split(tempName0[n][2], ' ') # Split event names whereever a space " " exists
            spkNames[n] = tempName1[n][0] # get event names 

        	#################   Create New Spike Names   ########################

            channel=getattr(hf5.root.sorted_units,spkNames[n]).channel[:]
            if int(olchan[0]) == int(channel[0]):
                char = char+1
            else:
                char = 97
            olchan = channel
            l = chr(char)
    
        	#################   Get Spike Timestamps   ########################

            spiketime=getattr(hf5.root.sorted_units,spkNames[n]).times[:]
            tempSPK = [] # For holding interger type spike times
            for a in range(len(spiketime)): #For all spike times
                tempSPK.append(spiketime[a].item()) #Convert numpy int64 to int
            data["Neurons"].append({ # Append data to the data variable in JSON format
                    'SPK%02d%s' % (int(channel[0]), l) : tempSPK
                    })

        	#################   Get Spike Waveforms   ########################
    
            waveform=getattr(hf5.root.sorted_units,spkNames[n]).waveforms[:]
            tempWave = [] # For holding 32 point waveforms
            waveforms = [] # For holding float type waveforms
            for a in range(len(waveform)): # For all waveforms
                tempWave = (list(waveform[a][0::10])) # Keep only every 10th point starting with the first point
                tempPoint = [] # For holding float type waveforms
                for b in range(len(tempWave)): # For all points in each waveform
                    tempPoint.append(tempWave[b].item()) # Convert numpy float64 to float
                waveforms.append(tempPoint) # Append each float waveform to list    
            data["Waveforms"].append({ # Append data to the data variable in JSON format
                    'SPK%02d%s_wf' % (int(channel[0]), l) : list(waveforms)
                    })
    	# Close the hdf5 file
        hf5.close()
        	#################   Export Data to JSON File   ########################
        try:
            infofile= [file for file in os.listdir(filedir+'/'+os.path.splitext(hdf5_name)[0]) if file.endswith('.info') ]
            if len(infofile)>1:
                sys.exit('Why do you have more than one infofile in this sorted folder?')
            elif len(infofile)==0:
                sys.exit("If you don't have an infofile, you should be using an older version of the postprocess script")
            else:
                file_iso['Recording Type']=config_handler.rec_info(filedir+'/'+os.path.splitext(hdf5_name)[0]+'/'+infofile[0])
                shutil.copy(filedir+'/'+os.path.splitext(hdf5_name)[0]+'/'+infofile[0],'R:/Autobots Roll Out/'+UserName+'/Info_Files')
        except Exception as e:
            print(e)
        with open("R:\\Autobots Roll Out\\%s\\JSON_Files\\%s.json" % (UserName, hdf5_name[:-3]), "w") as write_file:
            json.dump(data, write_file)
        print("JSON File Created")
        file_iso['Post-Process Date']=str(date.today())
        if not os.path.isfile("R:\\Autobots Roll Out\\"+UserName+'/Info_Files/Isolation Info.csv'):
            file_iso.to_csv("R:\\Autobots Roll Out\\"+UserName+'/Info_Files/Isolation Info.csv')
        else: 
            all_iso=pd.read_csv("R:\\Autobots Roll Out\\"+UserName+'/Info_Files/Isolation Info.csv')
            all_iso=all_iso.append(file_iso,ignore_index=True)
            for column in all_iso.columns:
                if 'Unnamed' in column:
                    all_iso=all_iso.drop(columns=[column])
            all_iso.to_csv("R:\\Autobots Roll Out\\"+UserName+'/Info_Files/Isolation Info.csv')
try: shutil.rmtree(temp_dir)
except: pass
        
            

    
    
