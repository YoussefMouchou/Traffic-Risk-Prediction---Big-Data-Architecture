# Traffic Risk Prediction — Big Data Architecture

This repository contains the implementation of a **Big Data architecture** designed to predict **traffic risks in real-time**, using technologies such as **Hadoop**, **Kafka**, **Spark**, **Spark Streaming**, **PostgreSQL**, **Streamlit**, and **Grafana**. The system processes mobility data and leverages **machine learning models** trained with **Spark MLlib** to forecast traffic conditions.

---

## 🚀 Project Overview

The goal of this project is to build a **scalable Big Data pipeline** for real-time traffic risk prediction. It:

- Collects and ingests real-time mobility data via Kafka.
- Processes and analyzes data with Spark and Spark Streaming.
- Applies a pre-trained ML model to detect traffic risk levels.
- Stores predictions in PostgreSQL.
- Visualizes insights in **Streamlit dashboards** and **Grafana panels**.

---

## 🧱 Architecture

The architecture consists of the following components:

- **Kafka**: Ingests real-time mobility data into the `mobility_data` topic.
- **HDFS**: Stores raw/processed data and trained ML models.  
  Example paths:  
  - `/data/mobility/processed/`  
  - `/models/traffic_model_optimized/`
- **Spark**: Performs batch processing and model training.
- **Spark Streaming**: Consumes Kafka data, applies the ML model, and sends predictions to PostgreSQL.
- **PostgreSQL**: Stores predictions in a table (`traffic_predictions`).
- **Streamlit & Grafana**: Visualize real-time predictions, trends, and performance metrics.

---

## 🔄 Data Flow

1. Raw CSV data (`smart_mobility_dataset.csv`) is converted to **Parquet**, partitioned by year/month/day.
2. A **Kafka Producer** pushes data to the `mobility_data` topic.
3. **Spark Streaming** consumes the topic, applies the ML model, and writes predictions to PostgreSQL.
4. **Dashboards** (Streamlit/Grafana) query PostgreSQL to display real-time risk insights.

---

## 🧰 Technologies Used

| Tool           | Purpose                          |
|----------------|----------------------------------|
| Hadoop (HDFS)  | Distributed storage              |
| Kafka          | Real-time data ingestion         |
| Spark          | Batch processing and ML training |
| Spark Streaming| Real-time processing             |
| PostgreSQL     | Prediction storage               |
| Streamlit      | Interactive dashboard            |
| Grafana        | Monitoring & analytics           |
| Python         | Core scripting & orchestration   |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YoussefMouchou/traffic-risk-prediction.git
cd traffic-risk-prediction
```

### 2. Set Up Hadoop

- Configure HDFS (e.g., on `hadoop-master:9000`)
- Upload raw data:

```bash
hdfs dfs -mkdir -p /data/mobility/raw/
hdfs dfs -put smart_mobility_dataset.csv /data/mobility/raw/
```

### 3. Set Up Kafka

Start **Zookeeper** and **Kafka**, then create a topic:

```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
bin/kafka-server-start.sh config/server.properties
bin/kafka-topics.sh --create --bootstrap-server localhost:9092 --replication-factor 1 --partitions 1 --topic mobility_data
```
### 4. 🧠 Prepare Spark
Ensure Spark 3.3.0 is installed.

Download the Kafka-Spark connector:
```
wget https://repo1.maven.org/maven2/org/apache/spark/spark-sql-kafka-0-10_2.12/3.3.0/spark-sql-kafka-0-10_2.12-3.3.0.jar
```

### 5. Set Up PostgreSQL

```sql
CREATE DATABASE traffic_db;
\c traffic_db
CREATE TABLE traffic_predictions (
    latitude FLOAT,
    longitude FLOAT,
    vehicle_count INT,
    traffic_speed_kmh FLOAT,
    road_occupancy_percent FLOAT,
    traffic_light_state VARCHAR(10),
    weather_condition VARCHAR(20),
    accident_report INT,
    sentiment_score FLOAT,
    ride_sharing_demand INT,
    parking_availability INT,
    emission_levels_g_km FLOAT,
    energy_consumption_l_h FLOAT,
    traffic_condition VARCHAR(20),
    year INT,
    month INT,
    day INT,
    hour INT,
    is_weekend INT,
    prediction FLOAT
);
```
## ▶️ Running the Project
1. Convert Raw Data to Parquet
```bash
python3 convert_to_parquet.py
```
3. Train the ML Model
```bash
spark-submit --master local[*] train_model.py
```
3. Start Kafka Producer
```bash
python3 producer-kafka.py
```
4. Run Spark Streaming Job
```bash
spark-submit \
--master local[1] \
--driver-memory 512m \
--executor-memory 512m \
--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
spark-streaming-postgre.py
```
5. Launch Streamlit Dashboard
```bash
streamlit run dashboard.py
```
7. Set Up Grafana
```bash
Connect Grafana to PostgreSQL.
```

Create dashboards using the traffic_predictions table for live analytics.
---

## 📊 Dashboard & Monitoring

- **Streamlit**: Launch with `streamlit run app.py`
![image](https://github.com/user-attachments/assets/372dc70c-1677-42b8-92ee-41b82768f1e3)
![image](https://github.com/user-attachments/assets/ab7bdd49-e0a1-4aca-ad97-72749a6b723f)
![image](https://github.com/user-attachments/assets/62ee215b-4231-447f-a979-11f5d5e71132)
![image](https://github.com/user-attachments/assets/a8281d39-b099-4554-bc5a-37f8317a7cc2)
- **Grafana**: Connect to PostgreSQL and create panels based on `traffic_predictions` table
![image](https://github.com/user-attachments/assets/0b7a428c-78f2-4830-a840-15b8b635b93d)

---

---

## 📌 Future Work / Improvements

-Integrate deep learning models.
-Optimize performance with Docker/Kubernetes.
-Enhance scalability for larger datasets.

---

## 📬 Contact

For questions or collaboration, reach out at: mouchouyoussef@gmail.com]
