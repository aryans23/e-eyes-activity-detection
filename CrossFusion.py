#!/usr/bin/env python3

import numpy as np
import pyemd as emd

# activity_profiles	# L X (J+1) X T
# there are 1...L devices
# there are J activity profile for each L device
# plus 1 for the empty room


def get_normalized_weights(activity_profiles):
	"""
		Input: activity profiles 
		Output: normalized weights
	"""
	L, J = activity_profiles.shape[0], activity_profiles.shape[1]-1
	chi = np.zeros((L,J))
	for i in range(L):
		a0 = activity_profiles[i][0]	# empty room
		for j in range(1,J+1):
			a = activity_profiles[i][j]
			chi[i][j-1] = np.correlate(a0,a)[0]
	chi_comp = 1-chi 
	sum_chi_comp = np.sum(chi_comp, axis = 0)	# 1 x J
	weights = np.zeros((L,J))
	for i in range(L):
		for j in range(J):
			weights[i][j] = chi_comp[i][j]/sum_chi_comp[j]
	return weights


def get_D(activity_profiles, csi_hist):
	"""
		activity_profiles are L x (J+1) x T
	"""
	L, J = activity_profiles.shape[0], activity_profiles.shape[1]-1
	D = np.array(L,J)
	for i in range(L):
		for j in range(1,J+1):
			hist1 = activity_profiles[i][j].astype(float)
			hist2 = csi_hist.astype(float)
			gd = pairwise.pairwise_distances(hist1.reshape(-1,1),hist2.reshape(-1,1))
			D[i][j] = emd(hist1, hist2, gd)
	return D


def final_activity(weights,D):
	ml = np.multiply(weights,D)
	w_mul_D_sum_L = np.sum(ml,axis=0)
	return np.argmin(w_mul_D_sum_L)