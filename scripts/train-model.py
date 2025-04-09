#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, dayofweek, when
from pyspark.ml import Pipeline
from pyspark.ml.feature import (VectorAssembler, StringIndexer, OneHotEncoder, Imputer)
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import logging
import sys
import time

# Simplified logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger('TrafficModel')

def create_spark_session():
    """Create a memory-efficient Spark session"""
    return SparkSession.builder \
        .appName("TrafficPrediction") \
        .config("spark.driver.memory", "1g") \
        .config("spark.executor.memory", "1g") \
        .config("spark.sql.shuffle.partitions", "1") \
        .config("spark.default.parallelism", "1") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.inMemoryColumnarStorage.compressed", "true") \
        .getOrCreate()


def prepare_data(spark):
    """Load and prepare data with minimal operations"""
    logger.info("Loading and optimizing data...")
    cols = ["Timestamp", "Vehicle_Count", "Road_Occupancy_%", "Weather_Condition", "Traffic_Condition"]
    df = spark.read.parquet("hdfs://hadoop-master:9000/data/mobility/processed").select(*cols)

    df = df.withColumn("hour", hour("Timestamp")) \
           .withColumn("is_weekend", when(dayofweek("Timestamp").isin([1, 7]), 1).otherwise(0)) \
           .drop("Timestamp")  # Remove unused column immediately

    return df
def train_model(train_df):
    """Train model with reduced complexity"""
    logger.info("Starting model training...")
    numeric_cols = ["Vehicle_Count", "Road_Occupancy_%"]
    categorical_cols = ["Weather_Condition"]

    stages = [
        Imputer(inputCols=numeric_cols, outputCols=[f"{c}_imputed" for c in numeric_cols]),
        StringIndexer(inputCol="Traffic_Condition", outputCol="label"),
        StringIndexer(inputCol=categorical_cols[0], outputCol="weather_index"),
        OneHotEncoder(inputCol="weather_index", outputCol="weather_encoded", dropLast=True),
        VectorAssembler(inputCols=[f"{c}_imputed" for c in numeric_cols] + ["weather_encoded", "hour", "is_weekend"], outputCol="features"),
        RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=10, maxDepth=5, seed=42, subsamplingRate=0.7)
    ]

    pipeline = Pipeline(stages=stages)

    start = time.time()
    model = pipeline.fit(train_df)
    logger.info(f"Training completed in {(time.time()-start)/60:.1f} minutes")

    return model

def evaluate_model(model, test_df):
    """Evaluate model with essential metrics only"""
    logger.info("Evaluating model...")
    predictions = model.transform(test_df)

    evaluator = MulticlassClassificationEvaluator(labelCol="label")
    accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)
    f1 = evaluator.setMetricName("f1").evaluate(predictions)

    logger.info("\n=== MODEL PERFORMANCE ===")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1 Score: {f1:.4f}\n")

    return {"Accuracy": accuracy, "F1 Score": f1}
def main():
    spark = None
    try:
        spark = create_spark_session()
        df = prepare_data(spark)
        train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

        model = train_model(train_df)
        metrics = evaluate_model(model, test_df)

        model_path = "hdfs://hadoop-master:9000/models/traffic_model_optimized"
        model.write().overwrite().save(model_path)
        logger.info(f"Model saved to {model_path}")

        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1
    finally:
        if spark:
            spark.stop()

if __name__ == "__main__":
    start_time = time.time()
    exit_code = main()
    logger.info(f"Pipeline completed in {(time.time()-start_time)/60:.1f} minutes")
    sys.exit(exit_code)
