#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, year, month, dayofmonth
import logging
import sys

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Cr  e une session Spark avec configuration optimis  e"""
    return SparkSession.builder \
        .appName("CSV_to_Parquet") \
        .config("spark.driver.memory", "2g") \
        .config("spark.executor.memory", "2g") \
        .master("local[*]") \
        .getOrCreate()

def main():
    """Fonction principale"""
    spark = None
    try:
        spark = create_spark_session()
        logger.info("Session Spark initialis  e")

        # Lecture
        df = spark.read.csv(
            "hdfs://hadoop-master:9000/data/mobility/raw/smart_mobility_dataset.csv",
            header=True,
            inferSchema=True
        )
        logger.info(f"Donn  es lues: {df.count()} lignes")
        
        # Transformation
        df = df.withColumn("Timestamp", to_timestamp(col("Timestamp"))) \
               .withColumn("year", year(col("Timestamp"))) \
               .withColumn("month", month(col("Timestamp"))) \
               .withColumn("day", dayofmonth(col("Timestamp")))

        #  ^icriture
        output_path = "hdfs://hadoop-master:9000/data/mobility/processed"
        df.write \
          .partitionBy("year", "month", "day") \
          .mode("overwrite") \
          .parquet(output_path)

        logger.info(f"Donn  es sauvegard  es dans {output_path}")
        return 0

    except Exception as e:
        logger.error(f"Erreur: {str(e)}", exc_info=True)
        return 1
    finally:
        if spark:
            spark.stop()
            logger.info("Session Spark ferm  e")

if __name__ == "__main__":
    sys.exit(main())
