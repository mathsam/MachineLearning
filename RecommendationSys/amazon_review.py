import json
import random
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
from sklearn import metrics
import implicit
import matplotlib.pylab as plt
from sklearn.decomposition import TruncatedSVD

from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import svds

from utils import Timer, train_test_split, Evaluator, BaselinePredictor

data = {
    'userId': [],
    'movieId': [],
    'rating': []
}

with Timer() as io_time:

    with open(r".\MachineLearning\All_Amazon_Review.json\All_Amazon_Review.json", "r", encoding="UTF-8") as f:
        for l in f:
            record = json.loads(l)
            data['userId'].append(record["reviewerID"])
            data['movieId'].append(record["asin"])
            data['rating'].append(record["overall"])

    data = pd.DataFrame(data)

    # create the sparse ratings matrix of users and items
    customers = sorted(list(data['userId'].unique())) # Get our unique customers
    products = sorted(list(data['movieId'].unique())) # Get our unique products
    quantity = list(data['rating'])

    rows = data['userId'].astype(pd.CategoricalDtype(categories=customers, ordered=True)).cat.codes
    # Get the associated row indices
    cols = data['movieId'].astype(pd.CategoricalDtype(categories=products, ordered=True)).cat.codes


    train, test = train_test_split(rows.values, cols.values, quantity)
    del quantity, rows, cols
    train_sparse = sparse.csr_matrix((train[2], (train[0], train[1])), shape=(len(customers), len(products)))

print("IO done in %f" %io_time.interval)


alpha = 15
with Timer() as cython_als_t:
    user_vecs, item_vecs = implicit.alternating_least_squares((train_sparse*alpha).astype('double'),
                                                              factors=64,
                                                              regularization = 0.1,
                                                              iterations = 10,
                                                              use_gpu=False)
print(f"Time spent in implicit: {cython_als_t.interval}")

evaluator = Evaluator(test[0], test[1], test[2], threshold=3.0)
baseline_model = BaselinePredictor(train[1], train[2])
baseline_fpr, baseline_tpr, baseline_roc = evaluator.roc(lambda user, item: baseline_model.pred(item))

fpr, tpr, roc = evaluator.roc(lambda user, item: np.sum(user_vecs[user, :] * item_vecs[item, :]))
print("AUC: %f" %roc)

plt.clf()
plt.plot(baseline_fpr, baseline_tpr, label='baseline')
plt.plot(fpr, tpr, label='als')
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.legend()
plt.show()