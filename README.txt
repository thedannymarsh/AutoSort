#About
Copyright (C) 2019-2020 Patricia Di Lorenzo & the Di Lorenzo Lab
This python package is a semi-supervised spike sorting/clustering algorithm adapted from the methods and code described in <Katz Article>

This program is designed to sort spikes into single units from continous electrophysiological recordings. Its main method of input is pl2 files, however, one could use it on other filetypes if the data were preprocessed into the necessary format (.h5 file containing the continuous signal from each electrode to be analyzed). Due to the limitations of the api used to parse data from .pl2 files, if these files are used, this program must be used on a windows computer. Otherwise it may be run on a Linux system.

For more information or questions, contact Daniel Marshall (dmarshal at binghamton dot edu)

#Installation
The following describes installation for the primary functionality of the program (Sorting units from pl2 files and storing as .nex files). Accessory scripts and utilities are described further down.

-Autosort_Main.py: This is the workhorse of this program. This script will take pl2 electrophysiological recordings, package their data into h5 files, parse action potentials from the continuous signal, and perform automated sorting on these spikes. The output will be a folder containing plots and data associated with the sort. 

-Autosort_Post.py: This script utilizes a GUI, and allows the user to postprocess sorted files. The user may mark clusters as units, merge clusters, or split clusters. Once the user has marked all of the units for this file, the script packages the data into a .json file.

-Json2Nex.py: This batch script will convert all .json files from post-processing into .nex files.

-pypl2: This library contains all of the necessary modules to run these scripts.

#Configuration Parameters:

[Run Settings]
resort limit - This is an integer representing the maximum number of times the program will attempt to resort any channels that did not sort properly the first time due to errors (Default = 3)
minimum licks - This is an integer representing the minimum number of licks a file must have to be sorte (Default = 1000)
cores used - This is an integer representing the number of cpu cores that will be utilized in parallel processing (Default=8  Note: Do not enter more than your total number of cpu cores. Additionally, it is recommended that you have at least 4GB of ram per core entered)
weekday run - This is an integer representing the number of files that will be processed on weekday evenings, if run type is set to 'Auto' (Default = 2)
weekend run - This is an integer representing the number of files that will be processed on weekends (i.e. the program is being run on a friday), if run type is set to 'Auto' (Default = 8)
run type - This denotes whether this run will be done manually, or automatically using task scheduler. (Default='Auto', options are 'Auto' or 'Manual', without quotation marks)
manual run = This is an integer representing the number of files which will be processed if run type is set to 'Manual' (Default=2)

[Paths]
pl2 to-run path - Path containing pl2 files to be run
running path - Path where files will be moved for processing
results path - Output path
completed pl2 path - Path where pl2 files will be moved upon completion
use paths - Whether to use the paths above, or the "else path." Choose 1 to use the multiple paths option, choose 0 to run everything within else path (not recommended if multiple computers are running from the same path
else path - Path to use if "use paths" is 0

[Clustering]
max clusters - An integer representing the maximum cluster count to use for Gaussian Mixture Modeling (GMM). The program will always start at 3. (Default = 7)
max iterations - Integer. Maximum GMM iterations (Default = 1000)
convergence criterion - Float. GMM convergence criterion (Default = .0001)
random restarts - Integer. Number of random restarts for GMM (Default = 10)
l-ratio cutoff - Integer. If the L-ratio for every cluster is above this cutoff, isolation information will not be calculated

[Signal]
disconnect voltage - Float. Voltage cutoff for disconnected headstage noise (in uV) (Default = 1500)
max breach rate - Integer. Maximum rate of cutoff breaches per sec (Default = .2)
max breach count - Float. Maximum number of allowed seconds with at least 1 cutoff breach (Default = 10)
max breach avg. - Float. Maximum allowed average number of cutoff breaches per sec (Default = 20)
intra-cluster cutoff - Float. Intra-cluster waveform amplitude SD cutoff (Default = 3)

[Filtering]
low cutoff - Float. Lower frequency cutoff (Hz) (Default = 300)
high cutoff - Float. Upper frequency cutoff (Hz) (Default = 6000)

[Spike]
pre-time - Float. Time before spike minimum (ms) (Default = 0.2)
post-time Float. Time after spike minimum (ms) (Default = 0.6)
sampling rate - Int Electrode data sample rate (Hz) (Default = 40000)

[Std Dev]
spike detection - Float. Standard deviation cutoff below mean electrode signal to detect a putative spike (Default = 2.0)
artifact removal = Float. Standard deviation cutoff above mean electrode signal to detect artifact (Default = 10.0)

[PCA]
variance explained - Float. Percentage variance to be explained by principal components (pc) used in gaussian mixture modeling
use percent variance - Logical. Choose 1 to select PC spaces based on variance explained. Choose 0 to use a user specified number of pc spaces.
principal component n - If 

[Post Process]
reanalyze = 0
simple gmm = 1
image size = 70
temporary dir = C:\Users\DiLorenzoTech\tmp_python

[Version]
config version = 3

#Usage

#Accessory Scripts and Utilities

#References
DOI 10.25080/shinma-7f4c6e7-00e
https://doi.org/10.1016/j.neuroscience.2004.09.066
doi: 10.1523/JNEUROSCI.4053-11.2011