#!/usr/bin/env python3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_json, struct
from confluent_kafka import Producer
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
        .appName("Parquet_to_Kafka") \
        .config("spark.driver.memory", "512m") \
        .config("spark.executor.memory", "512m") \
        .config("spark.executor.memoryOverhead", "256m") \
        .config("spark.memory.fraction", "0.4") \
        .config("spark.memory.storageFraction", "0.3") \
        .master("local[1]") \
        .getOrCreate()

def create_kafka_producer():
    """Cr  e un producteur Kafka"""
    # Configuration Kafka
    kafka_bootstrap_servers = "localhost:9092"
    kafka_topic = "mobility_data"

    # Cr  ation du producteur
    producer = Producer({
        'bootstrap.servers': kafka_bootstrap_servers,
        'client.id': 'mobility_producer'
    })

    return producer, kafka_topic
def main():
    """Fonction principale"""
    spark = None
    try:
        spark = create_spark_session()
        logger.info("Session Spark initialis  e")

        # Lecture des donn  es Parquet
        input_path = "hdfs://hadoop-master:9000/data/mobility/processed"
        df = spark.read.parquet(input_path)
        logger.info(f"Donn  es lues: {df.count()} lignes")

        # Transformation des donn  es en JSON
        df_json = df.select(to_json(struct([col(c) for c in df.columns])).alias("value"))

        # Cr  ation du producteur Kafka
        producer, kafka_topic = create_kafka_producer()
        logger.info(f"Producteur Kafka cr     pour le topic {kafka_topic}")

        # Production des donn  es dans Kafka
        for row in df_json.collect():
            message = row.value.encode('utf-8')
            producer.produce(kafka_topic, value=message)
            logger.info(f"Message produit dans {kafka_topic}")

        producer.flush()
        logger.info("Production termin  e")

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