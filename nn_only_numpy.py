"""vh.NN_only_numpy

Performs nearest neighbor classification on iris flower dataset using only the
NumPy package for computation.  Creates a file and saves results to file in 
same directory as code and datasets.
"""

import sys
import os
import numpy as np

def main():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    output_dir = base_dir+'\\nn_output.txt'
    
    # Read in features of training data.
    train_feat_arr = np.loadtxt("iris-training-data.csv", 
                                delimiter = ",",usecols = (0,1,2,3))
    # Read in classification of training data.
    train_class_arr = np.loadtxt("iris-training-data.csv", 
                                 delimiter = ",",usecols = (4), dtype = "str")
    # Read in features of test data.
    test_feat_arr = np.loadtxt("iris-testing-data.csv",
                               delimiter = ",",usecols = (0,1,2,3))
    # Read in classification of test data.
    test_class_arr = np.loadtxt("iris-testing-data.csv",
                                delimiter =',', usecols = (4), dtype = 'str')

    # Determine index of nearest neighbor NN for each data point in test set.
    nn_index = np.sqrt(((test_feat_arr[:,np.newaxis] 
                         - train_feat_arr[np.newaxis,:])**2).sum(2)).argmin(1)

    # Create array containing the predicted classification of each test
    # example based on the classification of the nearest neighbor.
    predicted = np.array((train_class_arr[nn_index]))

    # Calculate the accuracy of the predicted classification.
    accuracy = np.sum(test_class_arr == predicted) / len(test_class_arr) * 100

    # Writes output of test classification, predicted classification,
    # and percent accurate to text file then closes file.
    out_hdr = '#, True, Predicted'
    with open(output_dir, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        print(out_hdr+"\n") 
        for x in range(len(predicted)):
            print("%d, %s, %s" % (x + 1, test_class_arr[x], predicted[x]))
        print('Accuracy: %.2f%%' % (accuracy))
        sys.stdout = original_stdout
        
if __name__ == "__main__":
    main() 
    