Machine Learning optical spectra from Random Phase Approximation calculations (ML_RPA)              
Code for reproducing the results of Ref. [1], i.e. to                     
1. Predict the frequency-resolved imaginary part of the dielectric function of a broad class of materials from their crystal structure using data from calculations based on the Random Phase Approximation.                        
2. Compare the performance of various learning strategies.     

Small sections of the graph creation algorithm were adapted from the code published with Ref. [2].           

SYSTEM REQUIREMENTS:

The code requires an NVidia GPU with CUDA support to run. We note that to run without a GPU, all ".cuda()" calls just need to be changed to ".cpu()".               
The code has been tested on a Windows 10 computer and inside a Linux Docker container (image: mcr.microsoft.com/devcontainers/python:1-3.12-bullseye) running on Windows 10.                 
The version numbers are the versions that were tested - other versions might work, but success is not guaranteed.                    

Python v3.12.3
Installed with the following packages (plus their dependencies) with miniconda:           
- pytorch v2.3.0 with CUDA v12.1
- pytorch_geometric v2.5.3
- pymatgen v2024.5.1
- pytorch-lightning v2.5.2
- bokeh v3.7.3
- umap-learn v0.5.9

Installation of all packages should finish in a few minutes, depending on your internet speed.                   
No further installation is required.                     

USAGE:

0. Download files from [Figshare link]. Notably, the file "calc.zip" is not necessary for this part of the code.
1. Place the downloaded RPA data (the contents of "database.zip") into a folder named database.
2. Place the downloaded state_dicts (the contents of "trained_models.zip") into a folder named trained_models.
3. Place the downloaded images (the contents of "images.zip") into a folder named images.
4. Run the GraphCreation_RPA notebook to convert the PyMatGen ComputedStructureEntries to Pytorch-Geometric graphs.
5. Run the TransferLearn notebook to evaluate the models and carry out the analysis to produce the figures from [1]
    Note: Code for training models from scratch is clearly indicated and provided as reference but not necessary to run for reproduction of any results. 
6. Run the periodic_table_plot notebook to generate the periodic table plots for the distribution of elements in the training and test sets.

As output, figures of the publication should appear in the folder "plots", and various values reported in the paper should be printed in the TransferLearn notebook.


REFERENCES:

[1] M. Grunert, M. Großmann and E. Runge, Machine learning climbs the Jacob’s Ladder of optoelectronic properties, submitted                  
[2] J. Schmidt, L. Pettersson, C. Verdozzi, S. Botti and M. Marques, Crystal graph attention networks for the prediction of stable materials, Sci. Adv. 7, eabi7948 (2021)
