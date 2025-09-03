# PySpark Cinematch (Movie Recommendation System)
This project implements a robust Movie Recommendation System leveraging PySpark for large-scale data processing and machine learning, PostgreSQL as the primary data store.

## Features

1) **Recommendation Engine:** Content based filtering powered by PySpark's MLli

2) **Scalable Data Pipeline:** Efficient ETL processes handling large-scale movie datasets

3) **Relational Database Management:** PostgreSQL ensuring data integrity 

4) **Cloud Deployment & Scalability:** ETL pipeline implemented and managed on AWS for high availability and scalability 

## Prerequisites

Before installation, ensure you have the following installed:

- Python 3.9+

- Java JDK 11 (required by PySpark)

- PostgreSQL (v14+ recommended)

- Git

- PostgreSQL JDBC driver (postgresql-42.x.x.jar)

- VS Code with Python extension

## Installation & Setup

### 1. Clone the Repository

- git clone https://github.com/Prajwal-Luitel/Movie-Recommendation-ETL etl

### 2. Create and Activate Virtual Environment


#### Create virtual environment
python3 -m venv venv

#### Activate on Linux/Mac
source venv/bin/activate

#### Activate on Windows (PowerShell)
venv\Scripts\activate

#### Activate on Windows (CMD)
venv\Scripts\activate.bat

### 3. Install PySpark

- git clone https://github.com/neotheobserver/pyspark-install.git pyspark
- cd pyspark
- chmod +x pyspark-installation.sh
 run it with bash command
- ./pyspark-installation.sh 3.3.1 1.12.99

### 4. Install Python Dependencies

- pip install pyspark psycopg2 spark-nlp==6.0.2

### 5. Set Up PostgreSQL JDBC Driver
Download the PostgreSQL JDBC driver (e.g., postgresql-42.x.x.jar)

- Place the JAR file into the PySpark installation's jars/ directory within your virtual environment:

- In venv/lib/python3.x/site-packages/pyspark/jars

This step is essential for enabling PySpark's database connectivity.

### 7. Copy Kaggle URL of Movie Recommendation Dataset  
The ETL pipeline is configured to automatically download the required Movie dataset 
https://www.kaggle.com/datasets/shubhamchandra235/imdb-and-tmdb-movie-metadata-big-dataset-1m
get the direct URL from the above link by right clicking the download:




- Note :- If the direct download link in the code expire change it to new one.

### 8. Running the Application
a. Execute ETL Pipeline (VS Code Debugging)
Using the pre-configured launch.json:

Before that change the path according to your setup.

- Open the project in VS Code

- Navigate to the Run and Debug panel (Ctrl+Shift+D)

- Select the desired ETL configuration from the dropdown menu

- Click the Run button to execute the ETL scripts sequentially

- Available Debug Configurations:

- ETL: Extraction

- ETL: Transform

- ETL: Load
