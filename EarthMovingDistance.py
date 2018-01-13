#!/usr/bin/env python3

import pandas as pd
import numpy as np
import os
from pyemd import emd
from sklearn.metrics import pairwise

class MovingVariance(object):
	""" Calculates moving variance """
	def __init__(self, C, window):
		super(MovingVariance, self).__init__()
		self.C_ = C
		self.window_ = window
		print("MovingVariance initialized")

	def get_moving_var(self):
		return pd.rolling_var(self.C_,self.window_)
	
	def get_cumulative_moving_variance(self,V):
		return np.sum(V,axis=1)
	
	# def get_moving_var(self):
	# 	m_var = []
	# 	for i in range(self.C_.shape[0]):
	# 		m_var.append(pd.Series(self.C_[i]).rolling(window = self.window_).var())
	# 	return m_var


class EarthMovingDistance(object):
	""" 
		EarthMovingDistance class
		Input: amplitude histograms
		Output: E matrix
	"""
	def __init__(self, amplitude_histograms):
		super(EarthMovingDistance, self).__init__()
		self.amplitude_histograms = amplitude_histograms
		self.E = []
		self.closest = [] # can be calculated from E as well 
		print("EarthMovingDistance classifier initialized")

	def _ground_distance(self, histogram1, histogram2):
		return pairwise.pairwise_distances(histogram1.reshape(-1,1),histogram2.reshape(-1,1))

	def _get_emd(self, histogram1, histogram2):
		""" 
			Input: histograms -> histogram of amplitude bins for single round
			Output: The Earth Mover Distance between two histograms
		"""
		histogram1 = histogram1.astype(float)
		histogram2 = histogram2.astype(float)
		gd = self._ground_distance(histogram1, histogram2)
		return emd(histogram1, histogram2, gd)

	def get_EMD_matrix(self):
		"""
			Input: cached amplitude_histograms
			Output: cached EMD matrix (R x R)
		"""
		print("Calculating EMD matrix for %d histograms" %(len(self.amplitude_histograms))) 
		for C_r in self.amplitude_histograms:
			E_r = []
			for C_i in self.amplitude_histograms:
				emd = self._get_emd(C_r,C_i)
				E_r.append(emd)
				# print("EMD calculated in current iteration as ", str(emd))
			E_r_np = np.array(E_r)
			# calculating the second smallest emd as smallest would be 0
			closest_activity = E_r_np.argsort()[1] 
			self.closest.append(closest_activity)
			self.E.append(np.array(E_r))
		print("EMD matrix calculated!")
		return np.array(self.E)

	def get_closest_activity(self):
		print("Calculating closest_activity")
		if not self.E:
			self.get_EMD_matrix()
		print("closest_activity calculated")
		return np.array(self.closest)

def test():
    histograms = []
    for amplitude_matrix in amplitude_matrices:
    	for i in amplitude_matrix:
    		freq = np.zeros(40)
    		for j in i:
    			freq[int(j)] += 1
    	histograms.append(freq)
    labels_np = np.array(utl.labels)
    print(utl.labels)
    emd = EarthMovingDistance(histograms)
    emd_matrix = emd.get_EMD_matrix()

### Testing ###
if __name__ == '__main__':
    print("Starting testing for EMD")
    test()