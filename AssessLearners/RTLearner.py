"""Author Liu Meihan"""

import numpy as np

class RTLearner(object):
    """ Random Tree learner
        This learner behaves exactly like DTLearner,
        except that the choice of feature to split on are made randomly
    """

    def __init__(self, leaf_size, verbose=False):
        self.leaf_size = leaf_size
        pass

    def addEvidence(self, dataX, dataY):
        """
        @summary: Add training data to learner
        @param dataX: X values of data to add
        @param dataY: the Y training values
        """
        if dataX.shape[0] == 1:
            self.dtree = np.array([0, dataY, -1, -1])
        elif np.std(dataY - dataY[0]) == 0:
            self.dtree = np.array([0, dataY, -1, -1])
        else:
            self.dtree = self.buildtree(dataX, dataY)

    def buildtree(self, dataX, dataY):
        # build and save the model
        observs = dataX.shape[0]
        features = dataX.shape[1]
        if observs <= self.leaf_size:
            dtree = np.array([[-1, np.mean(dataY), -1, -1]])
        elif len(set(dataY)) == 1:
            dtree = np.array([[-1, np.mean(dataY), -1, -1]])
        else:
            indexlist = range(features)
            np.random.shuffle(indexlist)
            feature = indexlist[0]
            i = 0
            while len(set(dataX[:, feature])) == 1:
                feature = indexlist[i]
                i += 1

            splitVal = np.mean(np.random.choice(dataX[:, feature], size = 2, replace = False))
            while splitVal == np.amax(dataX[:, feature]):
                splitVal = np.mean(np.random.choice(dataX[:, feature], size = 2, replace = False))
            splitdata = dataX[:, feature] <= splitVal
            ltree = self.buildtree(dataX[splitdata], dataY[splitdata])
            rtree = self.buildtree(dataX[~splitdata], dataY[~splitdata])
            root = np.array([[feature, splitVal, 1, ltree.shape[0] + 1]])
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
    print "Random Tree Learner"