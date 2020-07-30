#About
Copyright (C) 2019-2020 Patricia Di Lorenzo & the Di Lorenzo Lab
This python package is a semi-supervised spike sorting/clustering algorithm adapted from the methods and code described in <Katz Article>

This program is designed to sort spikes into single units from continous electrophysiological recordings. Its main method of input is pl2 files, however, one could use it on other filetypes if the data were preprocessed into the necessary format (.h5 file containing the continuous signal from each electrode to be analyzed). Due to the limitations of the api used to parse data from .pl2 files, if these files are used, this program must be used on a windows computer. Otherwise it may be run on a Linux system.

For more information or questions, contact Daniel Marshall (dmarshal at binghamton dot edu)

#Installation
The following describes installation for the primary functionality of the program (Sorting units from pl2 files and storing as .nex files). Accessory scripts and utilities are described further down.

-Autosort_Main.py: This is the workhorse of this program. This script will take pl2 electrophysiological recordings, package their data into h5 files, parse action potentials from the continuous signal, and perform automated sorting on these spikes. The output will be a folder containing plots and data associated with the sort. 

-Autosort_Post.py: 

-Json2Nex.py:

-pypl2:

#Configuration Settings

#Usage

#Accessory Scripts and Utilities

#References
DOI 10.25080/shinma-7f4c6e7-00e
https://doi.org/10.1016/j.neuroscience.2004.09.066
doi: 10.1523/JNEUROSCI.4053-11.2011