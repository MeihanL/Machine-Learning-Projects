"""Author Liu Meihan"""

import numpy as np

class DTLearner(object):
    """ Decision Tree learner """

    def __init__(self, leaf_size, verbose=False):
        """ leaf_size is the maximum number of samples to be aggregated at a leaf """
        self.leaf_size = leaf_size
        pass

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        The columns are the features and the rows are the individual example instances
        """
        if dataX.shape[0] == 1:
            self.dtree = np.array([0, dataY, -1, -1])
        elif np.std(dataY - dataY[0]) == 0:
            self.dtree = np.array([0, dataY, -1, -1])
        else:
            self.dtree = self.buildtree(dataX, dataY)

    def buildtree(self, dataX, dataY):
        # build and save the model
        if dataX.shape[0] <= self.leaf_size:
            dtree = np.array([[-1, np.mean(dataY), -1, -1]])
        else:
            correlation = np.corrcoef(np.transpose(dataX), dataY)
            feature = np.argmax(abs(correlation[:, -1][0:-1]))
            splitval = np.median(dataX[:, feature])
            splitdata = dataX[:, feature] <= splitval
            if dataX[splitdata].shape[0] == dataX.shape[0]\
                    or dataX[~splitdata].shape[0] == dataX.shape[0]:
                dtree = np.array([[-1, np.mean(dataY), -1, -1]])
            else:
                ltree = self.buildtree(dataX[splitdata], dataY[splitdata])
                rtree = self.buildtree(dataX[~splitdata], dataY[~splitdata])
                root = np.array([[feature, splitval, 1, ltree.shape[0] + 1]])
                leftTree = np.append(root, ltree, axis = 0)
                dtree = np.append(leftTree, rtree, axis = 0)
        return dtree

    def query(self, points):
        """
        @summary: estimate a set of test points given the model we built
        @param points: a numpy array with each row corresponding to a specific query
        @returns the estimated values according to the saved model
        """
        rows = points.shape[0]
        testy = np.ones(rows)
        for i in np.arange(rows):
            index = 0
            node = int(self.dtree[index, 0])
            while node != -1:
                if points[i, node] <= self.dtree[index, 1]:
                    index = index + int(self.dtree[index, 2])
                elif points[i, node] > self.dtree[index, 1]:
                    index = index + int(self.dtree[index, -1])
                node = int(self.dtree[index, 0])
            testy[i] = self.dtree[index, 1]
        return testy

if __name__ == "__main__":
    print "Decision Tree Learner"