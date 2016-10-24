import sys
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

def knn(k, problemType, final_decision):
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
			print sqrt(square_dist)
			eucl_dist.append(sqrt(square_dist))

		# finding top k neighbours
		sorted_dist_index = np.argsort(np.array(eucl_dist))
		for s in sorted_dist_index:
			print "%d %.12f" %(s, eucl_dist[s]),
		#print sorted_dist_index
		k_list = sorted_dist_index[:k]
		print "k list ---------------"
		print k_list;
		tempList = [eucl_dist[k_list[0]], eucl_dist[k_list[1]], eucl_dist[k_list[2]]]
		print tempList

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(df.values[nbrs][-1])
		print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			print "decision map"
			print decision_map
			for dec in k_decisions:
				decision_map[dec] += 1
				#if decision_map[dec] > maximum[1] : maximum = (dec, )
			print decision_map
			for dec in metadata['class'][1]:
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

def knn_select(k, problemType, final_decision):
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
			print sqrt(square_dist)
			eucl_dist.append(sqrt(square_dist))

		# finding top k neighbours
		sorted_dist_index = np.argsort(np.array(eucl_dist))
		for s in sorted_dist_index:
			print "%d %.12f" %(s, eucl_dist[s]),
		#print sorted_dist_index
		k_list = sorted_dist_index[:k]
		print "k list ---------------"
		print k_list;
		tempList = [eucl_dist[k_list[0]], eucl_dist[k_list[1]], eucl_dist[k_list[2]]]
		print tempList

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(df.values[nbrs][-1])
		print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			print "decision map"
			print decision_map
			for dec in k_decisions:
				decision_map[dec] += 1
				#if decision_map[dec] > maximum[1] : maximum = (dec, )
			print decision_map
			for dec in metadata['class'][1]:
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


def print_op(k, problemType, final_decision):
	correctly_classified = 0
	mean_error = 0.0
	print "k value : %d" %(k)
	
	if problemType == "classification":
		
		for i in range(len(final_decision)) :
			print "test values"
			print testDf.values[i]
			print "Predicted class : %s\tActual class : %s" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1

		print "Number of correctly classified instances : %d" %(correctly_classified)
		print "Total number of instances : %d" %(len(final_decision))
		print "Accuracy : %f" %(correctly_classified / float(len(final_decision)))

	elif problemType == "regression":
		for i in range(len(final_decision)) :
			print "Predicted value : %.6f\tActual value : %f" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1
			mean_error += float(abs(testDf.values[i][-1] - final_decision[i]))

		mean_error = mean_error / float(len(final_decision)) 

		print "Mean absolute error : %.16f" %(mean_error)
		print "Total number of instances : %d" %(len(final_decision))	



# main function
if __name__ == "__main__":

	# check if it is classification or regression
	if df.columns[-1] == "class":
		problemType = "classification"
	elif df.columns[-1] == "response":
		problemType = "regression"

	final_decisions = [[], [], []]
	# k = int(sys.argv[3])
	# knn(k, problemType, final_decision)
	# print_op(k, problemType, final_decision)

	klist = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
	for k in klist:
		knn(k, final_decision[i])

	