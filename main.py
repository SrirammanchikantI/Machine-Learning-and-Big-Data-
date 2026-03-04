from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.param import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.ml.param.shared import HasInputCol, HasOutputCol

# -----------------------------
# Spark Session Configuration
# -----------------------------
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("Kickstarter Full Distinction ML Pipeline") \
    .config("spark.driver.host", "127.0.0.1") \
    .config("spark.driver.bindAddress", "127.0.0.1") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "50") \
    .getOrCreate()

# -----------------------------
# Custom Transformer (Correct Implementation)
# -----------------------------
class LogGoalTransformer(
    Transformer,
    HasInputCol,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable
):

    def __init__(self, inputCol=None, outputCol=None):
        super(LogGoalTransformer, self).__init__()
        kwargs = {}
        if inputCol is not None:
            kwargs["inputCol"] = inputCol
        if outputCol is not None:
            kwargs["outputCol"] = outputCol
        if kwargs:
            self._set(**kwargs)

    def _transform(self, dataset):
        return dataset.withColumn(
            self.getOutputCol(),
            log1p(col(self.getInputCol()))
        )

# -----------------------------
# Load Parquet Dataset
# -----------------------------
df = spark.read.parquet("data/parquet/kickstarter")

df = df.select(
    "category",
    "country",
    "goal",
    "state",
    "launched_at",
    "deadline",
    "staff_pick"
)

# -----------------------------
# Feature Engineering
# -----------------------------
df = df.withColumn("duration", col("deadline") - col("launched_at"))
df = df.withColumn("staff_pick", col("staff_pick").cast("integer"))
df = df.dropna()

df = df.repartition(50)
df.cache()
df.count()

# -----------------------------
# Transformers
# -----------------------------
log_transformer = LogGoalTransformer(
    inputCol="goal",
    outputCol="goal_log"
)

category_indexer = StringIndexer(
    inputCol="category",
    outputCol="category_index"
)

country_indexer = StringIndexer(
    inputCol="country",
    outputCol="country_index"
)

assembler = VectorAssembler(
    inputCols=[
        "goal_log",
        "duration",
        "staff_pick",
        "category_index",
        "country_index"
    ],
    outputCol="features"
)

# -----------------------------
# Train/Test Split
# -----------------------------
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

evaluator = BinaryClassificationEvaluator(labelCol="state")

# =============================
# 1️⃣ Logistic Regression
# =============================
lr = LogisticRegression(featuresCol="features", labelCol="state")

pipeline_lr = Pipeline(stages=[
    log_transformer,
    category_indexer,
    country_indexer,
    assembler,
    lr
])

model_lr = pipeline_lr.fit(train_data)
pred_lr = model_lr.transform(test_data)
accuracy_lr = evaluator.evaluate(pred_lr)

print("Logistic Regression Accuracy:", accuracy_lr)

# =============================
# 2️⃣ Random Forest
# =============================
rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="state",
    numTrees=40,
    maxDepth=6,
    maxBins=200
)

pipeline_rf = Pipeline(stages=[
    log_transformer,
    category_indexer,
    country_indexer,
    assembler,
    rf
])

model_rf = pipeline_rf.fit(train_data)
pred_rf = model_rf.transform(test_data)
accuracy_rf = evaluator.evaluate(pred_rf)

print("Random Forest Accuracy:", accuracy_rf)

# =============================
# 3️⃣ Decision Tree
# =============================
dt = DecisionTreeClassifier(
    featuresCol="features",
    labelCol="state",
    maxDepth=8,
    maxBins=200
)

pipeline_dt = Pipeline(stages=[
    log_transformer,
    category_indexer,
    country_indexer,
    assembler,
    dt
])

model_dt = pipeline_dt.fit(train_data)
pred_dt = model_dt.transform(test_data)
accuracy_dt = evaluator.evaluate(pred_dt)

print("Decision Tree Accuracy:", accuracy_dt)

# =============================
# 4️⃣ GBT with Cross Validation
# =============================
gbt = GBTClassifier(
    featuresCol="features",
    labelCol="state"
)

pipeline_gbt = Pipeline(stages=[
    log_transformer,
    category_indexer,
    country_indexer,
    assembler,
    gbt
])

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth, [3, 4])
             .addGrid(gbt.maxIter, [20, 30])
             .addGrid(gbt.maxBins, [200])
             .build())

crossval = CrossValidator(
    estimator=pipeline_gbt,
    estimatorParamMaps=paramGrid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2
)

cv_model = crossval.fit(train_data)

pred_gbt = cv_model.transform(test_data)
accuracy_gbt = evaluator.evaluate(pred_gbt)

print("Cross-Validated GBT Accuracy:", accuracy_gbt)

# -----------------------------
# Results Summary
# -----------------------------
print("\n===== Model AUC Summary =====")
print(f"  Logistic Regression : {accuracy_lr:.4f}")
print(f"  Random Forest       : {accuracy_rf:.4f}")
print(f"  Decision Tree       : {accuracy_dt:.4f}")
print(f"  GBT (CrossVal)      : {accuracy_gbt:.4f}")

# -----------------------------
# Save Best Model
# -----------------------------
cv_model.write().overwrite().save("models/best_gbt_model")

print("Best model saved successfully.")

spark.stop()