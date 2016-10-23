import sys
import math as math
import numpy as np
import scipy.io.arff as arff
import pandas as pd
training_arff = arff.loadarff(open("wine_train.arff"))
(data, metadata) = training_arff
df =  pd.DataFrame(data)
test_arff = arff.loadarff(open("wine_test.arff"))
(testData, testMetadata) = test_arff
testDf = pd.DataFrame(testData)
training_arff_c = arff.loadarff(open("yeast_train.arff"))
(data_c, metadata_c) = training_arff_c
df_c =  pd.DataFrame(data_c)
test_arff_c = arff.loadarff(open("yeast_test.arff"))
(testData_c, testMetadata_c) = test_arff_c
testDf_c = pd.DataFrame(testData_c)