import random
import pandas as pd
import scipy.sparse as sparse
import numpy as np
from scipy.sparse.linalg import spsolve
from sklearn import metrics
import implicit
import matplotlib.pylab as plt

from utils import make_train, Timer, calc_mean_auc, train_test_split, Evaluator, BaselinePredictor
from als import implicit_weighted_ALS

retail_data = pd.read_excel(r"./RecommendationSys/Online Retail.xlsx") # This may take a couple minutes
cleaned_retail = retail_data.loc[pd.isnull(retail_data.CustomerID) == False]

item_lookup = cleaned_retail[['StockCode', 'Description']].drop_duplicates() # Only get unique item/description pairs
item_lookup.loc[:, 'StockCode'] = item_lookup.StockCode.astype(str) # Encode as strings for future lookup ease

cleaned_retail.loc[:,'CustomerID'] = cleaned_retail.CustomerID.astype(int) # Convert to int for customer ID
cleaned_retail = cleaned_retail[['StockCode', 'Quantity', 'CustomerID']] # Get rid of unnecessary info
grouped_cleaned = cleaned_retail.groupby(['CustomerID', 'StockCode']).sum().reset_index() # Group together
grouped_cleaned.Quantity.loc[grouped_cleaned.Quantity == 0] = 1 # Replace a sum of zero purchases with a one to
# indicate purchased
grouped_purchased = grouped_cleaned.query('Quantity > 0') # Only get customers where purchase totals were positive

# create the sparse ratings matrix of users and items
customers = list(np.sort(grouped_purchased.CustomerID.unique())) # Get our unique customers
products = list(grouped_purchased.StockCode.unique()) # Get our unique products that were purchased
quantity = list(grouped_purchased.Quantity) # All of our purchases

rows = grouped_purchased.CustomerID.astype(pd.CategoricalDtype(categories=customers, ordered=True)).cat.codes
# Get the associated row indices
cols = grouped_purchased.StockCode.astype(pd.CategoricalDtype(categories=products, ordered=True)).cat.codes

train, test = train_test_split(rows.values, cols.values, quantity)

evaluator = Evaluator(test[0], test[1], test[2])
baseline_model = BaselinePredictor(train[1], train[2])
baseline_fpr, baseline_tpr, baseline_roc = evaluator.roc(lambda user, item: baseline_model.pred(item))

train_sparse = sparse.csr_matrix((train[2], (train[0], train[1])), shape=(len(customers), len(products)))

alpha = 15
with Timer() as cython_als_t:
    user_vecs, item_vecs = implicit.alternating_least_squares((train_sparse*alpha).astype('double'),
                                                              factors=32,
                                                              regularization = 0.1,
                                                              iterations = 50)
print(f"Time spent in implicit: {cython_als_t.interval}")

svd_predictor = lambda user, item: np.sum(user_vecs[user, :] * item_vecs[item, :])
fpr, tpr, roc = evaluator.roc(svd_predictor)

plt.clf()
plt.plot(baseline_fpr, baseline_tpr, label='baseline')
plt.plot(fpr, tpr, label='als')
plt.xlabel('False positive')
plt.ylabel('True positive')
plt.legend()
plt.show()