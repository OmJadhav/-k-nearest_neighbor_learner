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

	# go throuogh testing data to find out class of each instance
	for test_data in testDf.values:
		# test_data is a row in test data
		
		# ecludian list
		eucl_dist = []
		k_list = []

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
		tempList = [eucl_dist[k_list[0]], eucl_dist[k_list[1]], eucl_dist[k_list[2]]]
		print tempList

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
	print len(final_decision)

	# print to the console
	correctly_classified = 0
	print "k value : %d" %(k)
	
	if problemType == "classification":
		
		for i in range(len(final_decision)) :
			print "Predicted class : %s\tActual class : %s" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1

		print "Number of correctly classified instances : %d" %(correctly_classified)
		print "Total number of instances : %d" %(len(final_decision))
		print "Accuracy : %f" %(correctly_classified / float(len(final_decision)))

	elif problemType == "regression":
		for i in range(len(final_decision)) :
			print "Predicted value : %s\tActual value : %s" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1
			mean_error += abs(testDf.values[i][-1] - final_decision[i])

		mean_error /= float(len(final_decision)) 

		print "Mean absolute error : %d" %(mean_error)
		print "Total number of instances : %d" %(len(final_decision))