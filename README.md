## About
Copyright (C) 2019-2020 Patricia Di Lorenzo & the Di Lorenzo Lab

This python repository is a semi-supervised spike sorting/clustering algorithm adapted from the methods and code described in Mukherjee, Wachutka, & Katz (2017).

This program is designed to sort spikes into single, isolated units from electrophysiological recordings. Its main method of input is pl2 files, however, one could use it on other filetypes if the data were preprocessed into the necessary format (.h5 file containing the continuous signal  or thresholded waveforms from each electrode to be analyzed). Due to the limitations of the api used to parse data from .pl2 files, if this filetype is used, the program must be run on a windows computer. Otherwise it may be run on a Linux system.

This code is distributed under the GNU General Public License v3.0 (GPLv3), for more information, see the LICENSE file from this repository

For more information or questions, please contact Daniel Marshall (dmarshal at binghamton dot edu)

## Installation

The following describes installation for the primary functionality of the program (Sorting units from pl2 files and storing as .nex files). Accessory scripts and utilities are described later.

- Autosort_Main.py: This is the workhorse of this program. This script will take .pl2 format electrophysiological recordings (and behavioral data), package their information into h5 files, parse action potentials if using continuous signal, and perform automated sorting on these spikes. The output will be a folder containing plots and data associated with the sort. 

- Autosort_Post.py: This script utilizes a GUI, and allows the user to postprocess sorted files. The user may mark clusters as units, merge clusters, or split clusters. Once the user has marked all of the units for each file, the script packages the data into a .json file.

- Json2Nex.py: This batch script will convert all .json files from post-processing into .nex files.

- pypl2: This library contains all of the necessary modules to run these scripts.

## Usage
1) Processing

    Once all your configuration paramters have been set, usage is as follows:
    - Gather your .pl2 files in the relevant directory
    - Run the script
    - The script will create h5 files and a folder of data. Keep both of these, as they are both needed for Post-Processing.

2) Analysis

    The primary folder used for analysis will be the 'superplots' folder. All of these plots may be found in the 'Plots' folder, but in superplots they are compiled for user convenience. Additionally, the .info file contains information about the sort run, and the clustering_results_compiled_isoi.xlsx file contains information about each cluster in the sort. To consider a unit isolated, the following criteria must be met:
	- 1 millisecond ISIs (Inter-Spike Intervals) must be less than or equal to .5%
	- The waveform must be cellular
	- The unit must be sufficiently separated, as determined by the mahalanobis distribution graph on the bottom right of the superplots. This is the mahalanobis distance between the cluster of interest and each other cluster, calculated with number of dimensions d, where d=(number of PC spaces used+2) (+2 because energy and waveform peak amplitude are also used in the GMM)
	- Additionally, a unit can be immediately considered NOT isolated if its L-Ratio is greater than .1 (L-Ratio is calculated as described in Schmitzer-Torbert et al. (2005))
    - The plot paths are structured as follows: ./superplots/5/4_clusters/Cluster_1.png, where 5 is the electrode number, 4_clusters is the solution (GMM looked for 4 different clusters), and Cluster_1 is the cluster of interest for the image.

    Note: If there is clearly something separable in the data, but nothing isolated, remember that the Post-Processing step will allow you to try splitting up clusters, or merging clusters.

    Important: If indicating multiple clusters on a single electrode as isolated units, they must ALWAYS be from the same solution

    When searching for isolated units, you want to start with the 7 cluster solution. If you find something that has potential to be a unit, or is an isolated unit, continue moving to lower numbered solutions. The goal is to use the lowest numbered solution that renders the unit isolated, while avoiding incorporating waveforms that do not belong into the unit.
    
    The clustering_results_compiled_isoi.xlsx file can help with this process. Every cluster for each solution and electrode will be indicated by one line, and colored as follows:
    	- No Highlight: Acceptable L-Ratio and ISIs
    	- Yellow Highlight: ISIs between .5 and 1.0% (exclusive), or high L-ratio (as indicated by cutoff)
    	- Orange Highlight: ISIs greater than or equal to 1.0%, with an acceptable L-Ratio, or ISI's between .5 and 1.0% (exclusive) AND a high L-ratio (as indicated by cutoff)
    	- Red Highlight: ISIs greater than or equal to 1.0% AND high L-ratio (as indicated by cutoff)
    	Note: L-Ratio cutoff is defined by a parameter in the Autosort configuration file

    Once you have identified all of the isolated units for a file (or potential units to be merged/split), you can move on to the postprocessing step


3) Post-Processing

    Use of this script (Autosort_Post.py) is moderated by a GUI, and as such, this step is relatively straightforward. Run the script, and you will be asked to navigate to the folder containing your h5 files. You will then be directed to choose unit information to indicate isolated units (you may also split or merge clusters in this step). Indicating a unit to be not isolated at any point during this process will return you to electrode selection without saving data. One you have finished indicating all units for a file, the results will be packaged as a .json file
   
   Note: This step requires the presence of both the .h5 files, as well as the output folders from the Processing step

4) Json to Nex conversion

Running this script will convert your .json files into .nex files for response analysis

## Pipeline
The sorting pipeline functions as follows:
- Pre-Processing: Continuous signal data (and behavioral data) or thresholded waveforms are extracted from .pl2 files and packaged into .h5 files.
- Processing: The electrode signal is analyzed in the processing step, which involves the following (for more detailed information, see Mukherjee et al (2017)):
	- The electrode signal is cleaned and filtered (continuous only)
	- Waveforms are extracted from this signal(continuous only)
	- Waveforms are cleaned (continuous only) 
	- Waveforms are normalized based on energy
	- Principal components are calculated
	- Gaussian mixture modeling is performed with several target component counts, using principal components, as well as waveform energy and amplitude
	- Spikes are fit to the gaussian model which has the best bayesian criterion
	- Plots displaying waveforms, ISIs, Isolation, Feature vs Feature, and Mahalanobis distances between clusters are generated
	- Data and clustering information is packaged into .npy files
- Post-Processing: The data, based on user selection, is packaged into .json files. After this step is complete, all .h5 files and data folders may be safely deleted (Keep a copy of your .pl2 files though :D)
- Nex Conversion: .json files are converted to .nex files for analysis.

## Accessory Scripts and Utilities
Autosort_Updater.py: Used to check if a new version of the autosort is available, and if so, updates all computers

create_h5_batch.py: Used to create h5 files from pl2 files, without any processing. Can be used if you lost the .h5 files at some point.

Resort_GUI: GUI based program to sort individual channels, rather than whole files. Useful if you need to rerun a file, but only a couple channels. Can be run as a batch script (all electrodes for all files are indicated, then the script runs successively on each file)

start_autosort.bat & start_autosort_conda_not_in_path.bat: Can be used in conjunction with Windows Task Scheduler to facilitate automated activation of the autosort script.

Computing_Cluster: This folder contains scripts to facilitate running the autosort on a remote computing cluster server.

## Configuration Parameters:
These parameters can be found in the Autosort_config.ini file, which the Autosort will by default place in: C:/Users/<username>/Documents/Autosort

##### Run Settings
- resort limit - This is an integer representing the maximum number of times the program will attempt to resort any channels that did not sort properly the first time due to errors (Default = 3)
- minimum licks - This is an integer representing the minimum number of licks a file must have to be sorte (Default = 1000)
cores used - This is an integer representing the number of cpu cores that will be utilized in parallel processing (Default=8  Note: Do not enter more than your total number of cpu cores. Additionally, it is recommended that you have at least 4GB of ram per core entered)
- weekday run - This is an integer representing the number of files that will be processed on weekday evenings, if run type is set to 'Auto' (Default = 2)
- weekend run - This is an integer representing the number of files that will be processed on weekends (i.e. the program is being run on a friday), if run type is set to 'Auto' (Default = 8)
- run type - This denotes whether this run will be done manually, or automatically using task scheduler. (Default='Auto', options are 'Auto' or 'Manual', without quotation marks)
- manual run = This is an integer representing the number of files which will be processed if run type is set to 'Manual' (Default=2)

##### Paths
- pl2 to-run path - Path containing pl2 files to be run
- running path - Path where files will be moved for processing
- results path - Output path
- completed pl2 path - Path where pl2 files will be moved upon completion
- use paths - Whether to use the paths above, or the "else path." Choose 1 to use the multiple paths option, choose 0 to run everything within else path (not recommended if multiple computers are running from the same path
- else path - Path to use if "use paths" is 0

##### Clustering
- max clusters - An integer representing the maximum cluster count to use for Gaussian Mixture Modeling (GMM). The program will always start at 3. (Default = 7)
- max iterations - Integer. Maximum GMM iterations (Default = 1000)
convergence criterion - Float. GMM convergence criterion (Default = .0001)
random restarts - Integer. Number of random restarts for GMM (Default = 10)
- l-ratio cutoff - Integer. If the L-ratio for every cluster is above this cutoff, isolation information will not be calculated (Default = .1)

##### Signal
- disconnect voltage - Float. Voltage cutoff for disconnected headstage noise (in uV) (Default = 1500)
- max breach rate - Integer. Maximum rate of cutoff breaches per sec (Default = .2)
- max breach count - Float. Maximum number of allowed seconds with at least 1 cutoff breach (Default = 10)
- max breach avg. - Float. Maximum allowed average number of cutoff breaches per sec (Default = 20)
- intra-cluster cutoff - Float. Intra-cluster waveform amplitude SD cutoff (Default = 3)

##### Filtering
- low cutoff - Float. Lower frequency cutoff (Hz) (Default = 300)
- high cutoff - Float. Upper frequency cutoff (Hz) (Default = 6000)

##### Spike
- pre-time - Float. Time before spike minimum (ms) (Default = 0.2)
- post-time Float. Time after spike minimum (ms) (Default = 0.6)
- sampling rate - Int Electrode data sample rate (Hz) (Default = 40000)

##### Std Dev
- spike detection - Float. Standard deviation cutoff below mean electrode signal to detect a putative spike (Default = 2.0)
- artifact removal = Float. Standard deviation cutoff above mean electrode signal to detect artifact (Default = 10.0)

##### PCA
- variance explained - Float. Percentage variance to be explained by principal components (pc) used in gaussian mixture modeling (Default = .95)
- use percent variance - Logical. Choose 1 to select PC spaces based on variance explained. Choose 0 to use a user specified number of pc spaces. (Default = 1)
- principal component n - Integer. Number of principal components to use for GMM if 'use percent variance' = 0 (Default = 5)

##### Post Process
- reanalyze - Logical. Choose 0 to be queried as to whether or not you want to go over previously done files. Choose 1 to automatically skip this step (Default = 0)
- simple gmm - Logical. Choose 0 to be asked to enter all GMM parameters. Choose 1 to only be asked for cluster count when re-clustering. The parameters from your config file will then be utilized. (Default = 1)
- image size - Integer representing the size of the image to be created for use in postprocessing. (Default = 70)
- temporary dir - Directory where images will be temporarily stored while post-processing

##### Version
- config version - indicates config version. Do not modify this value

## References
Mukherjee, Narendra & Wachutka, Joseph & Katz, Donald. (2017). Python meets systems neuroscience: affordable, scalable and open-source electrophysiology in awake, behaving rodents. 98-105.

Schmitzer-Torbert N, Jackson J, Henze D, Harris K, Redish AD. Quantitative measures of cluster quality for use in extracellular recordings. Neuroscience. 2005;131:1–11.

https://plexon.com/software-downloads/#software-downloads-SDKs (Plexon Python API)
