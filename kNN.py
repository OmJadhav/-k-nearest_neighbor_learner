import sys
import math as math
import numpy as np
import scipy.io.arff as arff
import pandas as pd
from math import sqrt

# reading training data
training_arff = arff.loadarff(open(str(sys.argv[1]), 'rb'))
(data, metadata) = training_arff
df =  pd.DataFrame(data)

# for test data set 
test_arff = arff.loadarff(open(str(sys.argv[2]), 'rb'))
(testData, testMetadata) = test_arff
testDf = pd.DataFrame(testData)

k = int(sys.argv[3])



# main function
if __name__ == "__main__":

	# check if it is classification or regression
	if df.columns[-1] == "class":
		problemType = "classification"
	elif df.columns[-1] == "response":
		problemType = "regression"

	final_decision = []

	# # go throuogh testing data to find out class of each instance
	# for test_data in testDf.values:
	# 	# test_data is a row in test data
	# 	# print test_data
	# 	# ecludian list
	# 	eucl_dist = []
		
	# 	square_dist = 0

	# 	# find ecludian distance
	# 	for train_data in df.values:
	# 		print train_data
	# 		# for i, j in zip(train_data[: -1], test_data[: -1]):
	# 		# 	square_dist = square_dist + (i - j)**2
	# 		# eucl_dist.append(sqrt(square_dist))


	# # print eucl_dist[0]

	# go throuogh testing data to find out class of each instance
	for test_data in testDf.values:
			# test_data is a row in test data
			# print test_data
			# ecludian list
		eucl_dist = []
		k_list = []
			
		# square_dist = 0

		# find ecludian distance
		for train_data in df.values:
			#print train_data
			square_dist = 0
			for i, j in zip(train_data[: -1], test_data[: -1]):
				square_dist = square_dist + (i - j)**2
				# print square_dist
			eucl_dist.append(sqrt(square_dist))


		# finding top k neighbours
		sorted_dist_index = np.argsort(np.array(eucl_dist))
		#print len(sorted_dist_index)
		k_list = sorted_dist_index[:k]
		# print eucl_dist
		#print len(eucl_dist)
		# print k_list

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(df.values[nbrs][-1])
		# print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			for dec in k_decisions:
				decision_map[dec] += 1
				#if decision_map[dec] > maximum[1] : maximum = (dec, )
			for dec in decision_map:
				if decision_map[dec] > maximum[1] : maximum = (dec, decision_map[dec])
			# append to final decision list	
			final_decision.append(maximum[0])

		elif problemType == "regression":
			maximum = 0
			for dec in k_decisions:
				maximum = maximum + dec
			maximum = maximum / float(k)
			# append to final decision list
			final_decision.append(maximum)
			# TO DO : while printing format the float to 6 decimal digits print("%.6f" % maximum)


	print final_decision

		




