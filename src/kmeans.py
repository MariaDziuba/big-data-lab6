import pathlib
import configparser
import os
import path
import sys
cur_dir = path.Path(__file__).absolute()
sys.path.append(cur_dir.parent.parent)
tmp_dir = os.path.join(cur_dir.parent.parent.parent, "tmp")
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import SparkSession
from preprocess import Preprocessor


spark = SparkSession.builder \
    .master("local") \
    .appName("KMeans") \
    .getOrCreate()

class KMeansClustering:

    def clustering(self, scaled_data):
        evaluator = ClusteringEvaluator(
            predictionCol='prediction',
            featuresCol='scaled_features',
            metricName='silhouette',
            distanceMeasure='squaredEuclidean'
        )

        for k in range(2, 10):
            kmeans = KMeans(featuresCol='scaled_features', k=k)
            model = kmeans.fit(scaled_data)
            predictions = model.transform(scaled_data)
            score = evaluator.evaluate(predictions)
            print(f'k = {k}, silhouette score = {score}')


def main():
    config = configparser.ConfigParser()
    config.read('config.ini')

    path_to_data = os.path.join(cur_dir.parent.parent, config['data']['small_openfoodfacts'])
    preprocessor = Preprocessor()

    assembled_data = preprocessor.load_dataset(path_to_data, spark)
    scaled_data = preprocessor.scale_assembled_dataset(assembled_data)
    kmeans = KMeansClustering()
    kmeans.clustering(scaled_data)

    spark.stop()


if __name__ == '__main__':
    main()