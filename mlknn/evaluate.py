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
                if self.predict_labels[i][j] != self.test_labels[i][j]:
                    not_equal_num += 1
            hanmming_loss = hanmming_loss + not_equal_num / self.labels_num

        hanmming_loss = hanmming_loss / self.test_num

        return hanmming_loss

    def one_error(self):
        test_data_num = self.predict_rf.shape[0]
        class_num = self.predict_rf.shape[1]

        num = 0
        one_error = 0
        for i in range(test_data_num):
            if sum(self.predict_labels[i]) != class_num and sum(self.test_labels[i]) != 0:
                MAX = -np.inf
                #print(len(self.predict_labels[i]))
                for j in range(len(self.predict_labels[i])):
                    if self.predict_labels[i][j] > MAX:
                        index = j
                        MAX = self.predict_labels[i][index]
                num += 1
                if self.test_labels[i][index] != 1:
                    one_error += 1
        return one_error / num

    def coverage(self):
        coverage = 0

        test_data_num = self.predict_rf.shape[0]

        for i in range(test_data_num):

            record_idx = 0

            index = np.argsort(self.predict_rf[i])
            #print(index)
            for k in range(len(index)):
                if self.test_labels[i][index[k]] == 1:
                    record_idx = k
                    break
            #record_idx = record_idx + 1

            coverage += record_idx

        coverage = coverage/test_data_num
        return coverage

    '''
    @:param 
    @:return rank_loss pair_data smaller, better
    '''
    def rank_loss(self):
        rank_loss = 0

        test_data_num = self.predict_rf.shape[0]
        #class_num = self.predict_rf.shape[1]

        for i in range(test_data_num):
            Y = []
            Y_ = []
            num = 0
            #store the Y  and Y_
            for j in range(len(self.test_labels[i])):
                if self.test_labels[i][j] == 1:
                    Y.append(j)
                else:
                    Y_.append(j)
            #Y * Y_ length
            #print(Y, Y_, )
            Y_and_Y_ = len(Y)*len(Y_)
            #print(Y_and_Y_)
            for p in Y:
                for q in Y_:
                    if self.predict_rf[i][p] <= self.predict_rf[i][q]:
                        num += 1
            rank_loss += num / Y_and_Y_
        rank_loss = rank_loss / test_data_num
        return rank_loss



    def avg_precison(self):
        pass
