import time
import json
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

from pyspark.sql.functions import row_number, monotonically_increasing_id, col, mean
from pyspark.sql import Window
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

spark = SparkSession \
    .builder \
    .appName("ALSDim16") \
    .getOrCreate()

start_time = time.time()

print("starting computation")

spark.sparkContext.setCheckpointDir('/tmp')

lines = spark.sparkContext.textFile("wasb:///data/All_Amazon_Review.json")


def extraction(row):
    p = json.loads(row)
    return Row(userId=p['reviewerID'], itemId=p['asin'], rating=float(p['overall']))


ratingsRDD = lines.map(lambda row: extraction(row))

ratings = spark.createDataFrame(ratingsRDD)

user_rdd = ratings.select("userId").distinct().rdd.map(lambda x: x["userId"]).zipWithIndex()

user_index = spark.createDataFrame(user_rdd, ["userId", "userIdx"])

item_rdd = ratings.select("itemId").distinct().rdd.map(lambda x: x["itemId"]).zipWithIndex()

item_index = spark.createDataFrame(item_rdd, ["itemId", "itemIdx"])

ratings_with_userIdx = ratings.join(user_index, on=['userId'], how='left')

ratings_with_user_and_item_Idx = ratings_with_userIdx.join(item_index, on=['itemId'], how='left')

## persisting this dataframe is the key:

# https://medium.com/@meltem.tutar/pyspark-under-the-hood-randomsplit-and-sample-inconsistencies-examined-7c6ec62644bc

ratings_with_user_and_item_Idx.persist()

(training, test) = ratings_with_user_and_item_Idx.randomSplit([0.99, 0.01])

# Build the recommendation model using ALS on the training data

# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics

als = ALS(maxIter=10, regParam=0.2, userCol="userIdx", itemCol="itemIdx", ratingCol="rating", rank=16,
          coldStartStrategy="drop")

model = als.fit(training)

end_time = time.time()

print("Time elapsed %f" % (end_time - start_time))

# Evaluate the model by computing the RMSE on the test data

predictions = model.transform(test)

# evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
#                                predictionCol="prediction")

# rmse = evaluator.evaluate(predictions)

# print("Root-mean-square error = " + str(rmse))

all_items_ave_rating = training.select(mean('rating')).collect()[0][0]

ave_rating = training.select('itemIdx', 'rating').groupby('itemIdx').avg('rating')

ave_rating_test = test.join(ave_rating, on=['itemIdx'], how='left').na.fill(all_items_ave_rating)

predictions.select('rating', 'prediction').repartition(1).write.csv(
    path="wasb:///data/amazon_ratings_prediction_full_dim16", header=True)

ave_rating_test.select('rating', 'avg(rating)').repartition(1).write.csv(
    path="wasb:///data/amazon_ratings_prediction_baseline_full_dim16", header=True)
