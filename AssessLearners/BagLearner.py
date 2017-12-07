"""Author Liu Meihan"""

import numpy as np

class BagLearner(object):
    """ Bootstrap Aggregating
        BagLearner can accept any learner as input and use it to generate a learner ensemble
        so long as the learner obeys the API defined
    """

    def __init__(self, learner, kwargs, bags, verbose):
        self.learners = []
        self.bags = bags
        self.verbose = verbose
        for i in range(bags):
            self.learners.append(learner(**kwargs)) # create the learners (as many as the # of bags)

    def addEvidence(self, dataX, dataY=None):
        for i in range(self.bags):
            observs = len(dataY)
            n = np.random.choice(dataX.shape[0], observs)  # sample data with replacement
            trainingX = dataX[n, :]
            trainingY = dataY[n]
            self.learners[i].addEvidence(trainingX, trainingY)
        if self.verbose:
            print self.learners
        return self.learners

    def query(self, points):
        predictY = []
        for learner in self.learners:
            predictY.append(learner.query(points))
        predictY = np.array(predictY)
        predictY = np.nanmean(predictY, axis = 0)
        if self.verbose:
            print predictY
        return predictY

if __name__=="__main__":
    print "Bootstrap aggregating - Bag Learner"