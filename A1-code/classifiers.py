import numpy as np


# You need to build your own model here instead of using well-built python packages such as sklearn

# from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import Perceptron
# You can use the models form sklearn packages to check the performance of your own models

class HateSpeechClassifier(object):
    """Base class for classifiers.
    """

    def __init__(self):
        pass

    def fit(self, X, Y):
        """Train your model based on training set
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
            Y {type} -- array of actual labels, such as an N shape array, where N is the nu,ber of sentences
        """
        pass

    def predict(self, X):
        """Predict labels based on your trained model
        
        Arguments:
            X {array} -- array of features, such as an N*D shape array, where N is the number of sentences, D is the size of feature dimensions
        
        Returns:
            array -- predict labels, such as an N shape array, where N is the number of sentences
        """
        pass


class AlwaysPreditZero(HateSpeechClassifier):
    """Always predict the 0
    """

    def predict(self, X):
        return [0] * len(X)


class NaiveBayesClassifier(HateSpeechClassifier):
    """Naive Bayes Classifier
    """
    def __init__(self):
        # Vectors holding the counts of all positive/negative words in vocabulary
        self.pos_vocab = None
        self.neg_vocab = None
        
    def fit(self, X, Y):
        # Initializing corpus vectors
        self.pos_vocab = np.zeros(len(X[0]))
        self.neg_vocab = np.zeros(len(X[0]))

        # Separate words into pos/neg vectors & count total instances of each word
        # For every document in training data
        for i in range(0, len(X)):
            # For every word/feature in a document
            for j in range(len(X[i])):
                if X[i][j] != 0:
                    if Y[i] == 1:
                        self.pos_vocab[j] += X[i][j]
                    else:
                        self.neg_vocab[j] += X[i][j]
        
        # Count total instances of pos/neg
        self.totalPos = np.sum(self.pos_vocab)
        self.totalNeg = np.sum(self.neg_vocab)
        self.numFeatures = len(X[0])
        # print("totalPos", totalPos, "totalNeg", totalNeg, "numFeatures", numFeatures)
        
    def predict(self, X):
        # Vector storing predicted labels for each document
        predictions = np.zeros(len(X))
        # Setting smoothing alpha
        alpha = 1
        
        # Naive Bayes Classification
        # For every document in test data
        for i in range(len(X)):
            # Calculate summation of marginal probability logs for features in test document
            posTerm = 0
            negTerm = 0
            for j in range(len(X[i])):
                # For num times a feature shows up in test document, add smoothed marginal probability
                for numInstances in range(int(X[i][j])):
                    posTerm += np.log((self.pos_vocab[j] + alpha) / (self.totalPos + alpha * self.numFeatures))
                    negTerm += np.log((self.neg_vocab[j] + alpha) / (self.totalNeg + alpha * self.numFeatures))

            # Add prior probability logs to terms
            # remove smoothing?
            posTerm += np.log((self.totalPos) / (self.totalPos + self.totalNeg))
            negTerm += np.log((self.totalNeg) / (self.totalPos + self.totalNeg))

            # print("posTerm", posTerm, "negTerm", negTerm)
            if posTerm > negTerm: predictions[i] = 1
            else: predictions[i] = 0
        
        return predictions

class LogisticRegressionClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        self.beta = []

    def fit(self, X, Y):

        #initializing beta to a 0 vector size of #features
        #beta is also known as the weight
        self.beta = np.zeros(len(X[0]))

        #epoch is # of iterations before converging
        #chosen manually for now
        self.epoch = 20

        #initializing output of regression to vector size #sentences
        self.predictedX = np.zeros(len(X))

        #set learning rate 
        self.alpha = 0.1

        #set lambda values, ie regularization parameter
        self.lamda = np.full(len(X[0]), 0)

        #while not converged
        while self.epoch > 0:
            print("Epoch:", self.epoch)
            #for each sentence in X
            for i in range(0, len(X)):
                #Starting Regression
                #for each word in the sentence, add to sum of word * its weight 
                sum = 0
                for j in range(0, len(X[i])):
                    if X[i][j] != 0:
                        sum += X[i][j] * self.beta[j]
                #Binary logistic regression output is % chance of sentence labeled positive
                #calculate for each sentence and store in vector
                self.predictedX[i] = 1 / (1 + np.exp(-1 * sum))
                # print("actual:", Y[i], "predicted:", self.predictedX[i])

                #Starting Stochastic Gradient Descent
                #costGradient is vector of each features Loss * Count
                costGradient = (Y[i] - self.predictedX[i]) * X[i] 
                # print("total error:", np.sum(costGradient))

                #L2 Regularization term stored in vector for each feature
                regTerm = self.lamda * 2 * self.beta

                #update beta/weight vector
                self.beta += self.alpha * (costGradient - regTerm)
                # print("total weight:", np.sum(self.beta))

            totalLoss = np.sum(np.abs(Y - self.predictedX))
            print("loss", totalLoss)
            print("beta", self.beta)
            #update # epochs
            self.epoch -= 1

    def predict(self, X):
        #create 0 vector the length of #sentences
        predictedLabels = np.zeros(len(X))

        #for each sentence in X
        for i in range(0, len(X)):
            #Binary Logisitic Regression function
            sum = 0
            #for each word in the sentence, find summation
            for j in range(0, len(X[i])):
                if X[i][j] != 0:
                    sum += X[i][j] * self.beta[j]

            #if % from regression func is > 0.5, set label for sentence to 1
            #else set label for sentence to 0
            if (1 / (1 + np.exp(-1 * sum)) > 0.5):
                predictedLabels[i] = 1
            else:
                predictedLabels[i] = 0
        
        return predictedLabels


class PerceptronClassifier(HateSpeechClassifier):
    """Logistic Regression Classifier
    """

    def __init__(self):
        # Add your code here!
        raise Exception("Must be implemented")

    def fit(self, X, Y):
        # Add your code here!
        raise Exception("Must be implemented")

    def predict(self, X):
        # Add your code here!
        raise Exception("Must be implemented")


# you can change the following line to whichever classifier you want to use for bonus
# i.e to choose NaiveBayes classifier, you can write
# class BonusClassifier(NaiveBayes):
class BonusClassifier(PerceptronClassifier):
    def __init__(self):
        super().__init__()
