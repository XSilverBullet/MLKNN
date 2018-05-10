import  numpy as np
import operator

'''
get the k nearest neighbors index and distances

@:param 
    sample, (1, n)
    dataset (m , n) m is counts of samples
    k is real value
    
@:return
    sorted_dist_indicies (1, k)
    distances (1, k)
'''
def knn(sample, dataset, k):
    dataset_size = dataset.shape[0]

    #cal distance
    diffMat = np.tile(sample, (dataset_size, 1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sorted_dist_indicies = distances.argsort()

    if k > datasize:
        return None, None
    #print(sorted_dist_indicies)
    # class_count = {}
    # for i in range(k):
    #     vote_label = labels[sorted_dist_indicies[i]]
    #     class_count[vote_label] = class_count.get(vote_label, 0) + 1
    #
    #
    # sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_dist_indicies[:k], distances[:k]

if __name__=="__main__":
    #test
   datasize = 30
   dataset = np.random.rand(datasize*2).reshape(datasize,2)
   #labels = np.random.randint(1, 3, datasize)
   # print(labels)
   # print(dataset)
   index, distances = knn(dataset[0], dataset, 4)
   print(index)
   print(distances)

