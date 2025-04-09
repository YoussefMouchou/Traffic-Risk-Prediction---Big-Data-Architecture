#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, to_timestamp, hour, when, dayofweek
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
import logging
import sys

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_spark_session():
    """Cr  e une session Spark pour le streaming"""
    return SparkSession.builder \
        .appName("TrafficStreaming") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .getOrCreate()

def load_model(spark):
    """Charge le mod  le entra  n  """
    model_path = "hdfs://hadoop-master:9000/models/traffic_model_optimized"
    return PipelineModel.load(model_path)

def main():
    spark = None
    try:
        spark = create_spark_session()
        logger.info("Session Spark initialis  e")

        # Chargement du mod  le
        model = load_model(spark)
        logger.info("Mod  le charg  ")
        # D  finition du sch  ma des donn  es
        schema = StructType([
            StructField("Timestamp", StringType(), True),
            StructField("Latitude", FloatType(), True),
            StructField("Longitude", FloatType(), True),
            StructField("Vehicle_Count", IntegerType(), True),
            StructField("Traffic_Speed_kmh", FloatType(), True),
            StructField("Road_Occupancy_%", FloatType(), True),
            StructField("Traffic_Light_State", StringType(), True),
            StructField("Weather_Condition", StringType(), True),
            StructField("Accident_Report", IntegerType(), True),
            StructField("Sentiment_Score", FloatType(), True),
            StructField("Ride_Sharing_Demand", IntegerType(), True),
            StructField("Parking_Availability", IntegerType(), True),
            StructField("Emission_Levels_g_km", FloatType(), True),
            StructField("Energy_Consumption_L_h", FloatType(), True),
            StructField("Traffic_Condition", StringType(), True),
            StructField("year", IntegerType(), True),
            StructField("month", IntegerType(), True),
            StructField("day", IntegerType(), True)
        ])

        # Configuration Kafka
        kafka_bootstrap_servers = "localhost:9092"
        kafka_topic = "mobility_data"

        # Lecture des donn  es depuis Kafka
        df_stream = spark.readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", kafka_bootstrap_servers) \
            .option("subscribe", kafka_topic) \
            .option("startingOffsets", "earliest") \
            .load() \
            .select(from_json(col("value").cast("string"), schema).alias("data")) \
            .select("data.*")
        # Pr  paration des donn  es pour le mod  le
        df_stream = df_stream.withColumn("Timestamp", to_timestamp("Timestamp")) \
                             .withColumn("hour", hour("Timestamp")) \
                             .withColumn("is_weekend", when(dayofweek("Timestamp").isin([1, 7]), 1).otherwise(0)) \
                             .drop("Timestamp")

        # Application du mod  le
        predictions = model.transform(df_stream)

        # Select relevant columns for PostgreSQL
        predictions_to_save = predictions.select(
            col("Latitude").alias("latitude"),
            col("Longitude").alias("longitude"),
            col("Vehicle_Count").alias("vehicle_count"),
            col("Traffic_Speed_kmh").alias("traffic_speed_kmh"),
            col("Road_Occupancy_%").alias("road_occupancy_percent"),
            col("Traffic_Light_State").alias("traffic_light_state"),
            col("Weather_Condition").alias("weather_condition"),
            col("Accident_Report").alias("accident_report"),
            col("Sentiment_Score").alias("sentiment_score"),
            col("Ride_Sharing_Demand").alias("ride_sharing_demand"),
            col("Parking_Availability").alias("parking_availability"),
            col("Emission_Levels_g_km").alias("emission_levels_g_km"),
            col("Energy_Consumption_L_h").alias("energy_consumption_l_h"),
            col("Traffic_Condition").alias("traffic_condition"),
            "year", "month", "day", "hour", "is_weekend",
            "prediction"
        )

        # Write to PostgreSQL using .jdbc() to append
        def write_to_postgres(df, epoch_id):
            df.write.jdbc(
                url="jdbc:postgresql://172.18.0.2:5432/traffic_db",
                table="traffic_predictions",
                mode="append",
                properties={
                    "user": "mobility_user",
                    "password": "2005",
                    "driver": "org.postgresql.Driver"
                }
            )

        query = predictions_to_save.writeStream \
            .foreachBatch(write_to_postgres) \
            .outputMode("append") \
            .start()

        logger.info("Streaming d  marr  , writing to PostgreSQL")
        query.awaitTermination()

        return 0

    except Exception as e:
        logger.error(f"Erreur: {str(e)}")
        return 1
    finally:
        if spark:
            spark.stop()
            logger.info("Session Spark ferm  e")

if __name__ == "__main__":
    sys.exit(main())