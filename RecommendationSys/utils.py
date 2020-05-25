import collections
from collections import defaultdict
import time
import random
import numpy as np
from sklearn import metrics
import scipy.sparse as sparse


def make_train(ratings, pct_test = 0.2):
    '''
    This function will take in the original user-item matrix and "mask" a percentage of the original ratings where a
    user-item interaction has taken place for use as a test set. The test set will contain all of the original ratings,
    while the training set replaces the specified percentage of them with a zero in the original ratings matrix.

    parameters:

    ratings - the original ratings matrix from which you want to generate a train/test set. Test is just a complete
    copy of the original set. This is in the form of a sparse csr_matrix.

    pct_test - The percentage of user-item interactions where an interaction took place that you want to mask in the
    training set for later comparison to the test set, which contains all of the original ratings.

    returns:

    training_set - The altered version of the original data with a certain percentage of the user-item pairs
    that originally had interaction set back to zero.

    test_set - A copy of the original ratings matrix, unaltered, so it can be used to see how the rank order
    compares with the actual interactions.

    user_inds - From the randomly selected user-item indices, which user rows were altered in the training data.
    This will be necessary later when evaluating the performance via AUC.
    '''
    test_set = ratings.copy() # Make a copy of the original set to be the test set.
    test_set[test_set != 0] = 1 # Store the test set as a binary preference matrix
    training_set = ratings.copy() # Make a copy of the original data we can alter as our training set.
    nonzero_inds = training_set.nonzero() # Find the indices in the ratings data where an interaction exists
    nonzero_pairs = list(zip(nonzero_inds[0], nonzero_inds[1])) # Zip these pairs together of user,item index into list
    random.seed(0) # Set the random seed to zero for reproducibility
    num_samples = int(np.ceil(pct_test*len(nonzero_pairs))) # Round the number of samples needed to the nearest integer
    samples = random.sample(nonzero_pairs, num_samples) # Sample a random number of user-item pairs without replacement
    user_inds = [index[0] for index in samples] # Get the user row indices
    item_inds = [index[1] for index in samples] # Get the item column indices
    training_set[user_inds, item_inds] = 0 # Assign all of the randomly chosen user-item pairs to zero
    training_set.eliminate_zeros() # Get rid of zeros in sparse array storage after update to save space
    return training_set, test_set, list(set(user_inds)) # Output the unique list of user rows that were altered


def train_test_split(users, items, ratings, pct_test = 0.1, negative_sampling_ratio = 1.0):
    assert(len(users) == len(items))
    assert(len(users) == len(ratings))
    train = [[], [], []]
    test = [[], [], []]
    for user, item, rating in zip(users, items, ratings):
        which_list = train
        if random.random() < pct_test:
            which_list = test

        which_list[0].append(user)
        which_list[1].append(item)
        which_list[2].append(rating)

    neg_samples = 0
    user_item_sparse_m = sparse.csr_matrix((ratings, (users, items)), shape=(len(users), len(items)))
    while neg_samples < negative_sampling_ratio * len(users) * pct_test:
        user_idx = random.choice(users)
        item_idx = random.choice(items)
        if user_item_sparse_m[user_idx, item_idx] == 0:
            test[0].append(user_idx)
            test[1].append(item_idx)
            test[2].append(0)
            neg_samples += 1

    return train, test


class BaselinePredictor:

    def __init__(self, items, ratings):
        self.ave_rating = defaultdict(float)
        rating_count = defaultdict(int)
        for i, r in zip(items, ratings):
            self.ave_rating[i] += r
            rating_count[i] += 1

        for i in list(self.ave_rating.keys()):
            self.ave_rating[i] /= rating_count[i]

    def pred(self, items):
        if not isinstance(items, collections.Iterable):
            return self.ave_rating[items]

        ratings = []
        for i in items:
            ratings.append(self.ave_rating[i])

        return ratings

class Evaluator:

    def __init__(self, users, items, ratings, threshold = 0):
        """ratings above `threshold` will consider to be positive sample"""
        self.users = users
        self.items =items
        self.ratings = [r > 0 for r in ratings]

    def roc(self, predictor):
        preds = []
        for u, i in zip(self.users, self.items):
            preds.append(predictor(u, i))

        fpr, tpr, thresholds = metrics.roc_curve(self.ratings, preds)
        return fpr, tpr, metrics.auc(fpr, tpr)


class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start


def auc_score(predictions, test):
    '''
    This simple function will output the area under the curve using sklearn's metrics.

    parameters:

    - predictions: your prediction output

    - test: the actual target result you are comparing to

    returns:

    - AUC (area under the Receiver Operating Characterisic curve)
    '''
    fpr, tpr, thresholds = metrics.roc_curve(test, predictions)
    return metrics.auc(fpr, tpr)


def calc_mean_auc(training_set, altered_users, predictions, test_set):
    '''
    This function will calculate the mean AUC by user for any user that had their user-item matrix altered.

    parameters:

    training_set - The training set resulting from make_train, where a certain percentage of the original
    user/item interactions are reset to zero to hide them from the model

    predictions - The matrix of your predicted ratings for each user/item pair as output from the implicit MF.
    These should be stored in a list, with user vectors as item zero and item vectors as item one.

    altered_users - The indices of the users where at least one user/item pair was altered from make_train function

    test_set - The test set constucted earlier from make_train function



    returns:

    The mean AUC (area under the Receiver Operator Characteristic curve) of the test set only on user-item interactions
    there were originally zero to test ranking ability in addition to the most popular items as a benchmark.
    '''


    store_auc = [] # An empty list to store the AUC for each user that had an item removed from the training set
    popularity_auc = [] # To store popular AUC scores
    pop_items = np.array(test_set.sum(axis = 0)).reshape(-1) # Get sum of item iteractions to find most popular
    item_vecs = predictions[1]
    for user in altered_users: # Iterate through each user that had an item altered
        training_row = training_set[user,:].toarray().reshape(-1) # Get the training set row
        zero_inds = np.where(training_row == 0) # Find where the interaction had not yet occurred
        # Get the predicted values based on our user/item vectors
        user_vec = predictions[0][user,:]
        pred = user_vec.dot(item_vecs).toarray()[0,zero_inds].reshape(-1)
        # Get only the items that were originally zero
        # Select all ratings from the MF prediction for this user that originally had no iteraction
        actual = test_set[user,:].toarray()[0,zero_inds].reshape(-1)
        # Select the binarized yes/no interaction pairs from the original full data
        # that align with the same pairs in training
        pop = pop_items[zero_inds] # Get the item popularity for our chosen items
        store_auc.append(auc_score(pred, actual)) # Calculate AUC for the given user and store
        popularity_auc.append(auc_score(pop, actual)) # Calculate AUC using most popular and score
    # End users iteration

    return float('%.3f'%np.mean(store_auc)), float('%.3f'%np.mean(popularity_auc))
    # Return the mean AUC rounded to three decimal places for both test and popularity benchmark