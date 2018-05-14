# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:20:55 2018
@author: Wei Sun
"""

import numpy as np

'''
evaluate class
five metric functions
hanmming loss, one error, rank loss, coverage, average precision
'''


class Evaluate(object):
    '''
    @:param predict_labels, test_labels, predict_rf
    @:return none
        predict_labels , test_labels = (test_num, sum of labels)
        predict_rf = (test_num, sum of labels) is the f function value
    '''

    def __init__(self, predict_labels, test_labels, predict_rf):
        self.predict_labels = predict_labels
        self.test_labels = test_labels
        self.predict_rf = predict_rf

        # predict label's size
        self.labels_num = predict_labels.shape[1]

        # testing instances' size
        self.test_num = predict_labels.shape[0]

    '''
    @:param none
    @:return hanmming_loss(float)
        hamming loss: smaller, better
        the symmetric difference
    '''

    def hanmming_loss(self):

        hanmming_loss = 0

        for i in range(self.test_num):
            not_equal_num = 0
            for j in range(self.labels_num):
                if self.predict_labels[i][j] != self.test_labels[i][j]:
                    not_equal_num += 1
            hanmming_loss = hanmming_loss + not_equal_num / self.labels_num

        hanmming_loss = hanmming_loss / self.test_num

        return hanmming_loss

    '''
    @:param None
    @:return one_error( float )
        one_error, smaller, better
    '''

    def one_error(self):
        test_data_num = self.predict_rf.shape[0]
        class_num = self.predict_rf.shape[1]

        num = 0
        one_error = 0
        for i in range(test_data_num):
            if sum(self.predict_labels[i]) != class_num and sum(self.test_labels[i]) != 0:
                MAX = -np.inf
                # print(len(self.predict_labels[i]))
                for j in range(len(self.predict_labels[i])):
                    if self.predict_labels[i][j] > MAX:
                        index = j
                        MAX = self.predict_labels[i][index]
                num += 1
                if self.test_labels[i][index] != 1:
                    one_error += 1
        return one_error / num

    '''
    @:param none
    @:return coverage
        rate of coverage, smaller, better
    '''

    def coverage(self):
        coverage = 0

        test_data_num = self.predict_rf.shape[0]

        for i in range(test_data_num):

            record_idx = 0

            index = np.argsort(self.predict_rf[i])
            # print(index)
            for k in range(len(index)):
                if self.test_labels[i][index[k]] == 1:
                    record_idx = k
                    break
            # record_idx = record_idx + 1

            coverage += record_idx

        coverage = coverage / test_data_num
        return coverage

    '''
    @:param 
    @:return rank_loss(float)
        pair_data smaller, better
    '''

    def rank_loss(self):
        rank_loss = 0

        test_data_num = self.predict_rf.shape[0]
        # class_num = self.predict_rf.shape[1]

        for i in range(test_data_num):
            Y = []
            Y_ = []
            num = 0
            # store the Y  and Y_
            for j in range(len(self.test_labels[i])):
                if self.test_labels[i][j] == 1:
                    Y.append(j)
                else:
                    Y_.append(j)
            # Y * Y_ length
            # print(Y, Y_, )
            Y_and_Y_ = len(Y) * len(Y_)
            # print(Y_and_Y_)
            for p in Y:
                for q in Y_:
                    if self.predict_rf[i][p] <= self.predict_rf[i][q]:
                        num += 1
            rank_loss += num / Y_and_Y_
        rank_loss = rank_loss / test_data_num
        return rank_loss

    '''
    @:param none
    @:return avg_precision(float)
        average precision, larger, better
    '''

    def avg_precison(self):
        class_num = self.test_labels.shape[1]
        test_num = self.test_labels.shape[0]

        avg_precision = 0

        for i in range(test_num):
            s = 0
            # rankf = self.predict_rf[i]

            index = np.argsort(self.predict_rf[i])

            # yi_num = sum(self.test_labels[i])
            yi_num = class_num

            # print('yi count: ', yi_num)

            for j in range(class_num):
                num = 0
                # if self.test_labels[i][j] == 1:

                rf = np.argwhere(index == j)
                # print(float(rf)+1)

                for k in range(len(index)):
                    if rf > k:
                        num += 1
                s += num / (float(rf) + 1)

            avg_precision += s / yi_num

        avg_precision /= test_num

        return avg_precision
