#!/usr/bin/env python3

import numpy as np
import os
from Utilities import FileUtilities
from EarthMovingDistance import MovingVariance
from matplotlib import pyplot as plt
from CoarseActivityDetection import CoarseActivityDetection
# from InPlace import InPlace
from EarthMovingDistance import EarthMovingDistance
from CrossFusion import get_normalized_weights
from CrossFusion import get_D
from CrossFusion import final_activity


plt.figure(figsize=(18,6))
plt.title('Cumulative Moving Variance over time')
plt.xlabel('Time')
plt.ylabel('Cumulative Moving Variance')

path = '/Users/Apple/Documents/git-aryans/data/walking_loc_liv/'
utl = FileUtilities(path)

# input_file = '2-typing-parsed.csv'
# df = utl.read_csv(path+input_file)
# data = utl.get_data_matrix(df)
# C = data[:,1:]
# print(C)
# mv = MovingVariance(C,20)			# changing window from 1000 to 2
# V = mv.get_moving_var()
# print(V)
# print(V.shape)
# print(type(V))
# CMV = mv.get_cumulative_moving_variance(V)
# plt.plot(range(1, CMV.shape[0]+1) , CMV)

input_file = 'LivingToBed1-parsed.csv'
df = utl.read_csv(path+input_file)
data = utl.get_data_matrix(df)
C = data[:,1:]
print(C)
mv = MovingVariance(C,20)			# changing window from 1000 to 2
V = mv.get_moving_var()
print(V)
print(V.shape)
print(type(V))
CMV = mv.get_cumulative_moving_variance(V)
plt.grid()
plt.xticks(np.arange(0, CMV.shape[0], 50))
plt.plot(range(1, CMV.shape[0]+1) , CMV)

plt.show()

##########################################

# plt.figure(figsize=(12,6))
# plt.title('Cumulative Moving Variance over time')
# plt.xlabel('Time')
# plt.ylabel('Cumulative Moving Variance')
# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_small/'
# utl = FileUtilities(path)
# input_file_sitdown = 'input_161219_siamak_sitdown_1.dat.csv'
# df_sitdown = utl.read_csv(path+input_file_sitdown)
# data_sitdown = utl.get_data_matrix(df_sitdown)
# C_sitdown = data_sitdown[:,1:91]
# mv_sitdown = MovingVariance(C_sitdown,1000)
# V_sitdown = mv_sitdown.get_moving_var()
# print(V_sitdown.shape)
# print(type(V_sitdown))
# CMV_sitdown = mv_sitdown.get_cumulative_moving_variance(V_sitdown)
# plt.plot(range(1, CMV_sitdown.shape[0]+1) , CMV_sitdown)


# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_cur/'
# utl = FileUtilities(path)
# # input_file_run = 'data_small/input_161219_siamak_run_1.dat.csv'
# input_file_run = 'data_cur/1-empty-parsed.csv'
# df_run = utl.read_csv(input_file_run)
# data_run = utl.get_data_matrix(df_run)
# C_run = data_run[:,1:91]
# mv_run = MovingVariance(C_run,1000)
# V_run = mv_run.get_moving_var()
# print(V_run.shape)
# print(type(V_run))
# CMV_run = mv_run.get_cumulative_moving_variance(V_run)
# print(CMV_run)
# plt.plot(range(1, CMV_run.shape[0]+1) , CMV_run)
# plt.show()

##################

# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_small/'
# files = os.listdir(path)
# utl = FileUtilities(path)
# data_matrices = []
# for file in files:
# 	df = utl.read_csv(path+file)
# 	data = utl.get_data_matrix(df)
# 	data_matrices.append(data)
# # print(data_matrices)
# cad = CoarseActivityDetection(data_matrices,450)
# cad.get_cmv_for_all_files()
# # cad.plot_cmvs()
# cad.get_max_variance()
# cad.plot_max_cmvs()

# cad.get_cmv_for_all_files()

##################

# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_small/'
# utl = FileUtilities(path)
# amplitude_matrices = utl.get_amplitude_matrices()
# print(amplitude_matrices[0].shape)
# freq = np.zeros(40)
# for i in amplitude_matrices[0]:
# 	for j in i:
# 		freq[int(j)] += 1
# print(freq)
# plt.figure(figsize=(8,3))
# plt.title('Amplitude bins and counts')
# plt.xlabel('Bins')
# plt.ylabel('Amplitude Counts')
# # plt.grid(True)
# pos = np.arange(freq.shape[0])
# pos_x = np.arange(0,freq.shape[0],5)
# width = 0.7
# ax = plt.axes()
# ax.set_xticks(pos_x)
# ax.set_xticklabels(pos_x)
# # plt.bar(pos, freq, width)
# plt.bar(pos, freq, width,color='green')
# plt.show()

##################

# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_input/'
# utl = FileUtilities(path)
# amplitude_matrices = utl.get_amplitude_matrices()
# print("Number of matrices read = ", len(amplitude_matrices))

# histograms = []
# for amplitude_matrix in amplitude_matrices:
# 	for i in amplitude_matrix:
# 		freq = np.zeros(40)
# 		for j in i:
# 			freq[int(j)] += 1
# 	histograms.append(freq)

# labels_np = np.array(utl.labels)
# print(utl.labels)
# emd = EarthMovingDistance(histograms)
# emd_matrix = emd.get_EMD_matrix()
# closest_activity = emd.get_closest_activity()
# predicted = labels_np[closest_activity]
# print("______________________ EMD matrix start ______________________")
# print(emd_matrix)
# print("______________________ EMD matrix ends _______________________")
# print("___________________ closest_activity start ___________________")
# print(predicted)
# print("___________________ closest_activity ends ____________________")

##################

# path = '/Users/Apple/Documents/git-aryans/e-eyes-activity-detection/data_small/'
# utl = FileUtilities(path)
# amplitude_matrices = utl.get_amplitude_matrices()
# print("Number of matrices read = ", len(amplitude_matrices))
# labels_np = np.array(utl.labels)

# histograms = []
# for amplitude_matrix in amplitude_matrices:
# 	for i in amplitude_matrix:
# 		freq = np.zeros(40)
# 		for j in i:
# 			freq[int(j)] += 1
# 	histograms.append(freq)

# for n, freq in enumerate(histograms):
# 	ax = plt.subplot(len(histograms), 1, n+1)
# 	plt.title(labels_np[n])
# 	plt.xlabel('Bins')
# 	plt.ylabel('Amplitude Counts')
# 	# plt.grid(True)
# 	width = 0.7
# 	pos = np.arange(freq.shape[0])
# 	plt.bar(pos, freq, width)

# plt.show()



