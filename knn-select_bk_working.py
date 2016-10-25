import sys
import collections
import math as math
import numpy as np
import scipy.io.arff as arff
import pandas as pd
from math import sqrt

def knn(k, problemType, final_decision, df, testDf, metadata):
	# go throuogh testing data to find out class of each instance
	for test_data in testDf.values:
		# test_data is a row in test data
		
		# ecludian list
		eucl_dist = collections.OrderedDict()
		k_list = []

		# find ecludian distance
		for counter, train_data in zip(range(len(df.values)), df.values):
			#print train_data
			square_dist = 0
			for i, j in zip(train_data[: -1], test_data[: -1]):
				square_dist = square_dist + (i - j)**2
				# print square_dist
			eucl_dist[counter] = sqrt(square_dist)

		# finding top k neighbours
		sorted_dist_index = sorted(eucl_dist, key = eucl_dist.get)
		k_list = sorted_dist_index[:k]

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(df.values[nbrs][-1])
		#print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			for dec in k_decisions:
				decision_map[dec] += 1
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

def knn_select(k, problemType, final_decision, df, testDf, metadata):
	# go throuogh testing data to find out class of each instance
	for drop_index, test_data in zip(range(len(testDf.values)), testDf.values):
		# test_data is a row in test data
		
		# ecludian list
		eucl_dist = collections.OrderedDict()
		k_list = []
		one_record_found = 0

		# find ecludian distance
		tempDf = df.drop(df.index[drop_index])
		for counter, train_data in zip(range(len(tempDf.values)), tempDf.values):
			#print train_data
			square_dist = 0
			for i, j in zip(train_data[: -1], test_data[: -1]):
				square_dist = square_dist + (i - j)**2
				# print square_dist
			eucl_dist[counter] = sqrt(square_dist)

		# finding top k neighbours
		sorted_dist_index = sorted(eucl_dist, key = eucl_dist.get)
		k_list = sorted_dist_index[:k]

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(tempDf.values[nbrs][-1])
		#print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			for dec in k_decisions:
				decision_map[dec] += 1
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

def knn_select1(k, problemType, final_decision, df, testDf, metadata):
	# go throuogh testing data to find out class of each instance
	for test_data in testDf.values:
		# test_data is a row in test data
		
		# ecludian list
		eucl_dist = collections.OrderedDict()
		k_list = []
		one_record_found = 0

		# find ecludian distance
		for counter, train_data in zip(range(len(df.values)), df.values):
			#print train_data
			if len(np.unique(test_data == train_data)) > 1:
				square_dist = 0
				for i, j in zip(train_data[: -1], test_data[: -1]):
					square_dist = square_dist + (i - j)**2
					# print square_dist
				eucl_dist[counter] = sqrt(square_dist)
			else:
				one_record_found = 1

		# finding top k neighbours
		sorted_dist_index = sorted(eucl_dist, key = eucl_dist.get)
		k_list = sorted_dist_index[:k]

		# assigning class/response to the current instance of test data
		k_decisions = []
		for nbrs in k_list:
			k_decisions.append(df.values[nbrs][-1])
		#print k_decisions

		# find the mejority class for classification
		if problemType == "classification":
			# code
			decision_map = {}
			maximum = ('', 0)
			for c in metadata['class'][1]:
				decision_map[c] = 0
			for dec in k_decisions:
				decision_map[dec] += 1
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


def print_op(k, problemType, final_decision, df, testDf):
	correctly_classified = 0
	mean_error = 0.0
	print "Best k value : %d" %(k)
	
	if problemType == "classification":
		
		for i in range(len(final_decision)) :
			print "Predicted class : %s\tActual class : %s" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1

		print "Number of correctly classified instances : %d" %(correctly_classified)
		print "Total number of instances : %d" %(len(final_decision))
		print "Accuracy : %.16f" %(correctly_classified / float(len(final_decision)))

	elif problemType == "regression":
		for i in range(len(final_decision)) :
			print "Predicted value : %.6f\tActual value : %f" %(final_decision[i], testDf.values[i][-1])
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1
			mean_error += float(abs(testDf.values[i][-1] - final_decision[i]))

		mean_error = mean_error / float(len(final_decision)) 

		print "Mean absolute error : %.16f" %(mean_error)
		print "Total number of instances : %d" %(len(final_decision))	


def return_accuracy1(k, problemType, final_decision, testDf):
	correctly_classified = 0
	accuracy = 0
	mean_error = 0

	if problemType == "classification":
		for i in range(len(final_decision)) :
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1
		accuracy = correctly_classified / float(len(final_decision))

	elif problemType == "regression":
		for i in range(len(final_decision)) :
			if testDf.values[i][-1] == final_decision[i]:
				correctly_classified += 1
			mean_error += float(abs(testDf.values[i][-1] - final_decision[i]))
		mean_error = mean_error / float(len(final_decision)) 
		accuracy = mean_error

	print "accuracy _____________"
	print accuracy
	return accuracy

def return_accuracy(k, problemType, final_decision, testDf):
	incorrectly_classified = 0
	accuracy = 0
	mean_error = 0

	if problemType == "classification":
		for i in range(len(final_decision)) :
			if testDf.values[i][-1] != final_decision[i]:
				incorrectly_classified += 1
		accuracy = incorrectly_classified 

	elif problemType == "regression":
		for i in range(len(final_decision)) :
			mean_error += float(abs(testDf.values[i][-1] - final_decision[i]))
		mean_error = mean_error / float(len(final_decision)) 
		accuracy = mean_error

	print "accuracy _____________"
	print accuracy
	return accuracy


def print_accuracy_for_all_k(problemType, klist, accuracy, best_k):
	if problemType == "classification":
		for i, k in zip(accuracy, klist):
			print "Number of incorrectly classified instances for k = %d : %d" %(k, i)
	elif problemType == "regression":
		for i, k in zip(accuracy, klist):
			print "Mean absolute error for k = %d : %.16f" %(k, i)
	

	
# main function
if __name__ == "__main__":

	# reading training data
	training_arff = arff.loadarff(open(str(sys.argv[1]), 'rb'))
	(data, metadata) = training_arff
	df =  pd.DataFrame(data)

	# for test data set 
	test_arff = arff.loadarff(open(str(sys.argv[2]), 'rb'))
	(testData, testMetadata) = test_arff
	testDf = pd.DataFrame(testData)
	# check if it is classification or regression
	if df.columns[-1] == "class":
		problemType = "classification"
	elif df.columns[-1] == "response":
		problemType = "regression"

	# final_decision = []
	# k = int(sys.argv[3])
	# knn(k, problemType, final_decision, df, testDf, metadata)
	# print_op(k, problemType, final_decision, df, testDf)

	final_decision = [[], [], []]
	accuracy = []
	# k = int(sys.argv[3])
	# knn(k, problemType, final_decision)
	# print_op(k, problemType, final_decision)

	klist = [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
	for i, k in zip(range(len(klist)), klist):
		print i
		print k
		knn_select(k, problemType, final_decision[i], df, df, metadata)
		tempList = final_decision[i]
		accuracy.append(return_accuracy(k, problemType, tempList, df))

	print final_decision
	print len(final_decision)
	print accuracy

	sorted_klist_index = sorted(range(len(accuracy)), key=lambda k: accuracy[k])
	print sorted_klist_index

	# print the accuracy
	best_k = klist[sorted_klist_index[0]]
	print_accuracy_for_all_k(problemType, klist, accuracy, best_k)

	# call knn for best k
	final_decision_best_k = []
	knn(best_k, problemType, final_decision_best_k, df, testDf, metadata)
	print_op(best_k, problemType, final_decision_best_k, df, testDf)

	# do knn for first of the 