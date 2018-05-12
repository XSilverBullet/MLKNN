import numpy as np
import scipy
from scipy.io import arff
from sklearn.model_selection import KFold
import knn as knn
import evaluate as ev
import pandas as pd

class mlknn(object):

    def __init__(self, train_data, train_labels, k, s):
        self.train_data = train_data  # dataset = (m , n)
        self.train_labels = train_labels  # labels = (m,l), l is sum of all the multi-label
        self.train_lables_num = train_labels.shape[1]
        self.train_data_num = train_data.shape[0]
        self.k = k  # k is the real value
        self.s = s  # s is laplace parameter
        self.PH1 = np.zeros((self.train_lables_num,))
        self.PH0 = np.zeros((self.train_lables_num,))
        self.PEH1 = np.zeros((self.train_lables_num, self.k + 1))
        self.PEH0 = np.zeros((self.train_lables_num, self.k + 1))

    '''
        @:param
            train_labels
            train_data
        @:raise 
            train and MAP
    '''

    def fit(self):

        print("start to training...")
        # cal ph0 , ph1
        for i in range(self.train_lables_num):
            y = 0
            for j in range(self.train_data_num):
                if self.train_labels[j][i] == 1:
                    y += 1
            self.PH1[i] = (self.s + y) / (self.s * 2 + self.train_data_num)
        self.PH0 = 1 - self.PH1

        # cal peh1m peh0
        for i in range(self.train_lables_num):
            c1 = np.zeros((self.k + 1,))
            c0 = np.zeros((self.k + 1,))
            for j in range(self.train_data_num):
                temp = 0
                knn_index, knn_distances = knn.knn(self.train_data[j], self.train_data, self.k + 1)
                knn_index = knn_index[1:]
                knn_distances = knn_distances[1:]
                for index in knn_index:
                    if self.train_labels[index][i] == 1:
                        temp += 1

                if self.train_labels[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1

            for l in range(self.k + 1):
                self.PEH1[i][l] = (self.s + c1[l]) / (self.s * (self.k + 1) + c1.sum())
                self.PEH0[i][l] = (self.s + c0[l]) / (self.s * (self.k + 1) + c0.sum())
        print("training finished...")

    def predict(self, test_data):
        self.rtl = np.zeros((test_data.shape[0], self.train_lables_num))

        test_data_num = test_data.shape[0]
        self.predict_labels = np.zeros((test_data_num, self.train_lables_num))
        for i in range(test_data_num):
            # get k nearest neighbors' index in train data
            knn_index, knn_distances = knn.knn(test_data[i], self.train_data, self.k)

            for j in range(self.train_lables_num):
                temp = 0

                for index in knn_index:
                    if self.train_labels[index][j] == 1:
                        temp = temp + 1
                y1 = self.PH1[j] * self.PEH1[j][temp]
                y0 = self.PH0[j] * self.PEH0[j][temp]

                self.rtl[i][j] = self.PH1[j] * self.PEH1[j][temp] / (
                        self.PH1[j] * self.PEH1[j][temp] + self.PH0[j] * self.PEH0[j][temp])
                if y1 > y0:
                    self.predict_labels[i][j] = 1
                else:
                    self.predict_labels[i][j] = 0
        #print(self.predict_labels)
        return self.predict_labels

    def evaluate(self, test_labels):

        evaluate = ev.Evaluate(self.predict_labels, test_labels, self.rtl)
        print("hanmming loss: ")
        print(evaluate.hanmming_loss())

        print("one error: ")
        print(evaluate.one_error())

        print("rank loss: ")
        print(evaluate.rank_loss())

        print("coverage: ")
        print(evaluate.coverage())



def get_train_test():

    print("start to  load train data...")
    data, meta = scipy.io.arff.loadarff('../yeast/yeast-train.arff')
    df = pd.DataFrame(data)

    X_train = df.iloc[:,:103].values
    y_train = df.astype('int').iloc[:,103:]
    print("load train data finished...")
    #print(type(y_train))
    #logging.log(type(y_train),msg=None)

    print("start to load test data...")
    data, meta = scipy.io.arff.loadarff("../yeast/yeast-test.arff")
    df = pd.DataFrame(data)
    X_test = df.iloc[:,:103].values
    y_test = df.astype('int').iloc[:,103:].values
    print("load test data finished...")
    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = get_train_test()
    # kf = KFold(n_splits=10, shuffle=True, random_state=2017)
# for tr_index, val_index in kf.split(X_train):
    mlknn = mlknn(X_train, y_train, 10, 1)
    mlknn.fit()
    labels = mlknn.predict(X_test)
    mlknn.evaluate(y_test)
