
import numpy as np
import scipy.io as sio
from sklearn.model_selection import KFold
import mlknn.knn as knn
import mlknn.evaluate as ev

class mlknn(object):

    def __init__(self, train_data, labels, k, s ):
        self.train_data = train_data #dataset = (m , n)
        self.labels = labels   #labels = (m,l), l is sum of all the multi-label
        self.lables_num = labels.shape[1]
        self.train_data_num = train_data[0]
        self.k = k             #k is the real value
        self.s = s             #s is laplace parameter
        self.PH1 = np.zeros((self.lables_num,))
        self.PH0 = np.zeros((self.lables_num,))
        self.PEH1 = np.zeros((self.lables_num, self.k + 1))
        self.PEH0 = np.zeros((self.lables_num, self.k + 1))




    def fit(self):

        for i in range(self.lables_num):
            y = 0
            for j in range(self.train_data_num):
                if self.labels[j][i] == 1:
                    y += 1
            self.PH1[i] = (self.s + y)/(self.s * 2 + self.train_data_num)
        self.PH0 = 1 - self.PH1


        for i in range(self.lables_num):
            c1 = np.zeros((self.k + 1,))
            c0 = np.zeros((self.k + 1,))
            for j in range(self.train_data_num):
                temp = 0
                knn_index,knn_distances = knn.knn(self.train_data[j], self.train_data, self.k+1)
                knn_index = knn_index[1:]
                knn_distances = knn_distances[1:]
                for index in knn_index:
                    if self.labels[index][i] == 1:
                        temp += 1

                if self.labels[j][i] == 1:
                    c1[temp] = c1[temp] + 1
                else:
                    c0[temp] = c0[temp] + 1


            for l in range(self.k + 1):
                self.PEH1[i][l] = (self.s + c1[l])/(self.s * (self.k + 1) + c1.sum())
                self.PEH0[i][l] = (self.s + c0[l])/(self.s * (self.k + 1) + c0.sum())

    def predict(self, test_data):
        self.rtl = np.zeros((test_data.shape[0], self.lables_num))

        test_data_num = test_data.shape[0]
        self.predict_labels = np.zeros((test_data_num, self.lables_num))
        for i in range(test_data_num):
            knn_index, knn_distances = knn.knn(self.test_data[i], self.train_data, self.k)

            for j in range(self.lables_num):
                temp=0
                y1=0
                y0=0
                for index in knn_index:
                    if self.labels[index][j] == 1:
                        temp = temp + 1
                y1 = self.PH1[j]*self.PEH1[j][temp]
                y0 = self.PH0[j]*self.PEH0[j][temp]

                self.rtl[i][j] = self.PH1[j]*self.PEH1[j][temp]/(self.PH1[j]*self.PEH1[j][temp] + self.PH0[j]*self.PEH0[j][temp])
                if y1 > y0:
                    self.predict_labels[i][j] = 1
                else:
                    self.predict_labels[i][j] = 0
        return self.predict_labels


    def evaluate(self, test_labels):

        evaluate = ev.Evaluate(self.predict_labels, test_labels, self.rtl)
        print("hanmming loss: ")
        print(evaluate.hanmming_loss())

        print("one error: ")
        print(evaluate.one_error())

if __name__=="__main__":
    a = np.random.random(100*2).reshape(100,2)
    print(a.shape[0],
    a.shape[1])