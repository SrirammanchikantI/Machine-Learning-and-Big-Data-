from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder \
    .appName("Kickstarter Data Engineering") \
    .config("spark.hadoop.fs.file.impl", "org.apache.hadoop.fs.LocalFileSystem") \
    .config("spark.hadoop.fs.file.impl.disable.cache", "true") \
    .getOrCreate()

# --------------------------
# Ingestion
# --------------------------
df = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .option("multiLine", True) \
    .option("escape", '"') \
    .option("quote", '"') \
    .csv("data/kickstarter_2022-2021_unique_blurbs.csv")

print("Initial Row Count:", df.count())

# --------------------------
# Data Validation
# --------------------------
df = df.filter(col("goal") > 0)
df = df.filter(col("state").isin(0,1))
df = df.dropna(subset=["category", "country"])

print("After Validation Row Count:", df.count())

# --------------------------
# Feature Engineering
# --------------------------
df = df.withColumn("duration", col("deadline") - col("launched_at"))

# --------------------------
# Persist in Memory
# --------------------------
df.persist()

# --------------------------
# Write as Partitioned Parquet
# --------------------------
df.write \
    .mode("overwrite") \
    .partitionBy("country") \
    .parquet("data/parquet/kickstarter")

print("Parquet file created successfully.")

df.unpersist()

spark.stop()