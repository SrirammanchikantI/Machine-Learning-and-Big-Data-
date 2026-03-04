from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("Kickstarter Big Data Project") \
    .getOrCreate()

print("Spark is working!")

spark.stop()