#!/usr/bin/env python3

import numpy as np
import os
from Utilities import FileUtilities
from matplotlib import pyplot as plt
from DynamicTimeWarping import DynamicTimeWarping

def main():
	'''
		Retuns after distinguishing walking path
	'''
	path = '/Users/Apple/Documents/gitRepo/ra-eeyes-activity-detection/data_not_inplace/'
	utl = FileUtilities(path)
	amplitude_matrices = utl.get_amplitude_matrices()
	print("Number of matrices read = ", len(amplitude_matrices))
	histograms = []
	for amplitude_matrix in amplitude_matrices:
		for i in amplitude_matrix:
			freq = np.zeros(40)
			for j in i:
				freq[int(j)] += 1
		histograms.append(freq)
	labels_np = np.array(utl.labels)
	print(utl.labels)
	walking_detect = DynamicTimeWarping(histograms)
	dtw_matrix = walking_detect.get_DTW_matrix()
	closest_path = walking_detect.get_closest_activity()
	path = labels_np[closest_path]
	print(dtw_matrix)
	print(path)

if __name__ == '__main__':
	main()