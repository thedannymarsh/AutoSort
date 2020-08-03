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
pl2 to-run path = R:\Dannymarsh Sorting Emporium\pl2_to_be_sorted
running path = R:\Dannymarsh Sorting Emporium\running_files
results path = R:\Dannymarsh Sorting Emporium\Results
completed pl2 path = R:\Dannymarsh Sorting Emporium\completed_pl2
use paths = 1
else path = 

[Clustering]
max clusters = 7
max iterations = 1000
convergence criterion = .0001
random restarts = 10
l-ratio cutoff = .1

[Signal]
disconnect voltage = 1500
max breach rate = .2
max breach count = 10
max breach avg. = 20
intra-cluster cutoff = 3

[Filtering]
low cutoff = 300
high cutoff = 6000

[Spike]
pre-time = .2
post-time = .6
sampling rate = 40000

[Std Dev]
spike detection = 2.0
artifact removal = 10.0

[PCA]
variance explained = .95
use percent variance = 1
principal component n = 5

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