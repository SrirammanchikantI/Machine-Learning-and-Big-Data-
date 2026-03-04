"""
test_pipeline.py
Unit and integration tests for the Kickstarter ML pipeline.
Run with: python tests/test_pipeline.py
"""

import os
import sys
import unittest

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def get_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("TestPipeline")
        .master("local[2]")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "4")
        .getOrCreate()
    )


class TestDataEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark()
        # Load the already-built Parquet file
        cls.df = cls.spark.read.parquet("data/parquet/kickstarter")

    def test_parquet_not_empty(self):
        self.assertGreater(self.df.count(), 0, "Parquet dataset should not be empty")

    def test_required_columns_present(self):
        required = {"category", "country", "goal", "state", "launched_at", "deadline", "staff_pick"}
        actual = set(self.df.columns)
        self.assertTrue(required.issubset(actual), f"Missing columns: {required - actual}")

    def test_goal_positive(self):
        negative = self.df.filter(col("goal") <= 0).count()
        self.assertEqual(negative, 0, "All goal values should be positive after validation")

    def test_state_binary(self):
        invalid = self.df.filter(~col("state").isin(0, 1)).count()
        self.assertEqual(invalid, 0, "state column should only contain 0 or 1")

    def test_no_null_key_columns(self):
        for c in ["category", "country", "goal", "state"]:
            nulls = self.df.filter(col(c).isNull()).count()
            self.assertEqual(nulls, 0, f"Column '{c}' should have no nulls after ingestion")

    def test_country_partitioning_exists(self):
        parquet_root = "data/parquet/kickstarter"
        subdirs = [d for d in os.listdir(parquet_root) if d.startswith("country=")]
        self.assertGreater(len(subdirs), 1, "Parquet should be partitioned by country")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class TestFeatureEngineering(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark()
        raw = cls.spark.read.parquet("data/parquet/kickstarter")
        cls.df = (
            raw.select("category", "country", "goal", "state",
                       "launched_at", "deadline", "staff_pick")
               .withColumn("duration", col("deadline") - col("launched_at"))
               .withColumn("goal_log",  log1p(col("goal")))
               .withColumn("staff_pick", col("staff_pick").cast("integer"))
               .dropna()
        )

    def test_duration_column_created(self):
        self.assertIn("duration", self.df.columns)

    def test_goal_log_column_created(self):
        self.assertIn("goal_log", self.df.columns)

    def test_goal_log_non_negative(self):
        negatives = self.df.filter(col("goal_log") < 0).count()
        self.assertEqual(negatives, 0, "log1p(goal) should always be >= 0")

    def test_no_nulls_after_feature_engineering(self):
        nulls = self.df.filter(
            col("duration").isNull() | col("goal_log").isNull()
        ).count()
        self.assertEqual(nulls, 0, "No nulls after feature engineering")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class TestMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.spark = get_spark()
        raw = cls.spark.read.parquet("data/parquet/kickstarter")
        cls.df = (
            raw.select("category", "country", "goal", "state",
                       "launched_at", "deadline", "staff_pick")
               .withColumn("duration", col("deadline") - col("launched_at"))
               .withColumn("goal_log",  log1p(col("goal")))
               .withColumn("staff_pick", col("staff_pick").cast("integer"))
               .dropna()
               .limit(5000)   # Use a small slice for fast tests
        )
        cls.train, cls.test = cls.df.randomSplit([0.8, 0.2], seed=42)

    def test_logistic_regression_pipeline_fits(self):
        cat_idx = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="keep")
        ctr_idx = StringIndexer(inputCol="country",  outputCol="country_index",  handleInvalid="keep")
        asm = VectorAssembler(
            inputCols=["goal_log", "duration", "staff_pick", "category_index", "country_index"],
            outputCol="features"
        )
        lr  = LogisticRegression(featuresCol="features", labelCol="state")
        pipeline = Pipeline(stages=[cat_idx, ctr_idx, asm, lr])

        model = pipeline.fit(self.train)
        preds = model.transform(self.test)
        self.assertIn("prediction", preds.columns)

    def test_evaluator_returns_valid_auc(self):
        cat_idx = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="keep")
        ctr_idx = StringIndexer(inputCol="country",  outputCol="country_index",  handleInvalid="keep")
        asm = VectorAssembler(
            inputCols=["goal_log", "duration", "staff_pick", "category_index", "country_index"],
            outputCol="features"
        )
        lr  = LogisticRegression(featuresCol="features", labelCol="state")
        pipeline = Pipeline(stages=[cat_idx, ctr_idx, asm, lr])

        model = pipeline.fit(self.train)
        preds = model.transform(self.test)
        evaluator = BinaryClassificationEvaluator(labelCol="state")
        auc = evaluator.evaluate(preds)
        self.assertGreater(auc, 0.5, "AUC should be above random baseline (0.5)")
        self.assertLessEqual(auc, 1.0, "AUC cannot exceed 1.0")

    def test_train_test_split_sizes(self):
        total = self.df.count()
        train_n = self.train.count()
        test_n  = self.test.count()
        self.assertEqual(train_n + test_n, total, "Train+test rows should equal total")
        self.assertAlmostEqual(train_n / total, 0.8, delta=0.05,
                               msg="Train split should be ~80%")

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()


class TestModelSerialization(unittest.TestCase):

    def test_gbt_model_saved(self):
        self.assertTrue(
            os.path.isdir("models/best_gbt_model/metadata"),
            "GBT model metadata directory should exist after training"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
