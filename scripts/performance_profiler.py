"""
performance_profiler.py
Strong & Weak Scaling Analysis for the Kickstarter ML Pipeline.

Strong scaling: fixed dataset size, vary executor memory / shuffle partitions.
Weak scaling: dataset fraction grows proportionally with partition count.

Run from workspace root:
    python scripts/performance_profiler.py
"""

import time
import json
import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, log1p
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_PATH = "data/parquet/kickstarter"

STRONG_PARTITIONS = [50, 100, 200, 400]
WEAK_FRACTIONS    = [0.25, 0.50, 0.75, 1.0]


def build_spark(app_name: str, shuffle_partitions: int = 200) -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .getOrCreate()
    )


def build_pipeline() -> Pipeline:
    category_indexer = StringIndexer(inputCol="category", outputCol="category_index", handleInvalid="keep")
    country_indexer  = StringIndexer(inputCol="country",  outputCol="country_index",  handleInvalid="keep")
    assembler = VectorAssembler(
        inputCols=["goal_log", "duration", "staff_pick", "category_index", "country_index"],
        outputCol="features"
    )
    rf = RandomForestClassifier(featuresCol="features", labelCol="state", numTrees=20, maxDepth=5, maxBins=200)
    return Pipeline(stages=[category_indexer, country_indexer, assembler, rf])


def prepare_df(spark: SparkSession, fraction: float = 1.0, partitions: int = 200):
    df = spark.read.parquet(PARQUET_PATH)
    if fraction < 1.0:
        df = df.sample(False, fraction, seed=42)
    df = (
        df.select("category", "country", "goal", "state", "launched_at", "deadline", "staff_pick")
          .withColumn("duration", col("deadline") - col("launched_at"))
          .withColumn("goal_log",  log1p(col("goal")))
          .withColumn("staff_pick", col("staff_pick").cast("integer"))
          .dropna()
          .repartition(partitions)
    )
    df.persist()
    df.count()  # materialise cache
    return df


def time_fit(df, partitions: int) -> dict:
    pipeline = build_pipeline()
    train, test = df.randomSplit([0.8, 0.2], seed=42)
    evaluator = BinaryClassificationEvaluator(labelCol="state")

    t0 = time.time()
    model = pipeline.fit(train)
    train_time = time.time() - t0

    t1 = time.time()
    preds = model.transform(test)
    auc = evaluator.evaluate(preds)
    infer_time = time.time() - t1

    return {
        "partitions": partitions,
        "row_count": df.count(),
        "train_time_s": round(train_time, 2),
        "infer_time_s": round(infer_time, 2),
        "auc": round(auc, 4),
    }


def strong_scaling_experiment(spark: SparkSession) -> list:
    print("\n--- Strong Scaling (fixed data, varying partitions) ---")
    results = []
    for p in STRONG_PARTITIONS:
        print(f"  partitions={p} ...", end=" ", flush=True)
        df = prepare_df(spark, fraction=1.0, partitions=p)
        r  = time_fit(df, p)
        df.unpersist()
        results.append(r)
        print(f"train={r['train_time_s']}s  AUC={r['auc']}")
    return results


def weak_scaling_experiment(spark: SparkSession) -> list:
    print("\n--- Weak Scaling (data size grows with partitions) ---")
    results = []
    for i, frac in enumerate(WEAK_FRACTIONS):
        partitions = STRONG_PARTITIONS[i]
        print(f"  fraction={frac:.2f}  partitions={partitions} ...", end=" ", flush=True)
        df = prepare_df(spark, fraction=frac, partitions=partitions)
        r  = time_fit(df, partitions)
        r["fraction"] = frac
        df.unpersist()
        results.append(r)
        print(f"rows={r['row_count']}  train={r['train_time_s']}s  AUC={r['auc']}")
    return results


def bottleneck_summary(strong: list, weak: list) -> dict:
    """Identify I/O vs compute bottleneck from timing ratios."""
    # If training time grows super-linearly with row count => compute-bound
    # If training time is flat with more partitions => I/O-bound
    train_times = [r["train_time_s"] for r in strong]
    speedup = [train_times[0] / t if t > 0 else 0 for t in train_times]
    return {
        "strong_speedups": speedup,
        "note": (
            "Super-linear speedup with partitions suggests compute bottleneck. "
            "Sub-linear or flat suggests shuffle / I/O bottleneck."
        )
    }


if __name__ == "__main__":
    spark = build_spark("PerformanceProfiler")

    strong_results = strong_scaling_experiment(spark)
    weak_results   = weak_scaling_experiment(spark)
    summary        = bottleneck_summary(strong_results, weak_results)

    report = {
        "strong_scaling": strong_results,
        "weak_scaling":   weak_results,
        "bottleneck_analysis": summary,
    }

    out_path = os.path.join(OUTPUT_DIR, "scaling_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nScaling report saved to {out_path}")
    spark.stop()
