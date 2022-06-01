# CS230
CS 230: CELL TYPE CLASSIFICATION VIA LEARNING OF SPATIOTEMPORAL ELECTRICAL SIGNATURES
Code and Files Most Applicable for Curious Graders:

testval_sets.txt and traintrainval_sets.txt are lists of retinal pieces appropriate for 
those set divisions. These are the lists that will be used to begin the features generation.

features_DLelec.py contains a number of new features I developed over the course of this 
project. Features are applied using celltable.

generateTrainFeatures.ipynb and generateTestFeatures.ipynb calculate all the features for
all the pieces in the respective sets and stores them in dataframes saved to pickle files.
Due to the large number of datasets and the large values of the EI, this had to be done in
batches and with EI being removed before saving.

loadtraintest.ipynb is where features are then put into arrays for examples and labels. 
This includes loading the dataframes, shuffling piece ids, distributing piece ids to the 
appropriate sets based on distribution, ensuring features are all the same length no matter
experimental protocol used, normalizing features based on training set. Creating the arrays. 
Shuffling the cells one final time.

Future updates for the prototypical network coming
