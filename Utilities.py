#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os

class FileUtilities(object):
	""" 
		Reads the CSI data 
	"""
	def __init__(self, path):
		super(FileUtilities, self).__init__()
		self.path = path
		self.data_matrices = []
		self.amplitude = []
		self.labels = []
		print("FileUtilities initialized")
		
	def read_csv(self,file):
		return pd.read_csv(file)

	def get_data_matrix(self,df):
		""" Returns the timestamps, amplitude and phase as a single matrix """
		return(df.as_matrix())

	def get_data_matrices(self):
		files = os.listdir(self.path)
		for file in files:
			df = self.read_csv(self.path+file)
			# label = str(str(file).split('-')[1])
			label = str(str(file).split('_')[3])
			self.labels.append(label)
			print("Reading " + str(file) + "...")
			data = self.get_data_matrix(df)
			self.data_matrices.append(data)
		print("FileUtilities::get_data_matrices read %d files" %(len(self.data_matrices)))
		return self.data_matrices

	def get_amplitude_matrices(self):
		if not self.data_matrices:
			self.get_data_matrices()
		for data_matrix in self.data_matrices:
			self.amplitude.append(np.array(data_matrix[:,1:]))		# for new data
			# self.amplitude.append(np.array(data_matrix[:,1:91]))
		return self.amplitude