
import numpy as np
import scipy.io as sci

class Evaluate(object):
    '''
        predict_labels , test_labels = (test_num, sum of labels)
    '''
    def __init__(self, predict_labels, test_labels, predict_rf):
        self.predict_labels = predict_labels
        self.test_labels = test_labels
        self.predict_rf = predict_rf

        self.labels_num = predict_labels.shape[1]
        self.test_num = predict_labels.shape[0]

    '''
        hamming loss: smaller, better
    '''
    def hanmming_loss(self):

        hanmming_loss = 0

        for i in range(self.test_num):
            not_equal_num = 0
            for j in range(self.labels_num):
                if self.predict_labels[i][j]!=self.test_labels[i][j]:
                    not_equal_num += 1
            hanmming_loss = hanmming_loss + not_equal_num / self.labels_num

        hanmming_loss = hanmming_loss/self.test_num

        return hanmming_loss

    def one_error(self):
        test_data_num = self.predict_rf.shape[0]
        class_num = self.predict_rf.shape[1]

        num = 0
        one_error = 0
        for i in range(test_data_num):
            if sum(self.predict_labels[i])!= class_num and sum(self.test_labels[i])!=0:
                MAX = -np.inf
                print(len(self.predict_labels[i]))
                for j in range(len(self.predict_labels[i])):
                    if self.predict_labels[i][j] > MAX:
                        index = j
                        MAX = self.predict_labels[i][index]
                num += 1
                if self.test_labels[i][index]!=1:
                    one_error += 1
        return one_error/num
