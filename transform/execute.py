import sys, os, time, sparknlp

from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql import functions as F

from sparknlp.base import DocumentAssembler, Finisher
from sparknlp.annotator import (
    Tokenizer,
    Normalizer as NLPNormalizer,
    LemmatizerModel,
    StopWordsCleaner,
)

from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    CountVectorizer,
    Normalizer as VectorNormalizer,
    BucketedRandomProjectionLSH,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utility.utility import setup_logging, format_time


def create_spark_session(logger, spark_config):
    """Initialize Spark session."""
    logger.info("Starting spark session")
    return (
        SparkSession.builder.master(f"spark://{spark_config['master_ip']}:7077")
        .appName("MovieDataTransform")
        .config("spark.driver.memory", spark_config["driver_memory"])
        .config("spark.executor.memory", spark_config["executor_memory"])
        .config("spark.executor.cores", spark_config["executor_cores"])
        .config("spark.executor.instances", spark_config["executor_instances"])
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:6.0.2")
        .getOrCreate()
    )


def load_and_clean(logger, spark, input_dir):
    """Stage 1: Load data, drop duplicates, remove nulls, replace nulls, save cleaned data."""

    logger.info("Loading data in dataframes")

    movie_df = spark.read.csv(
        os.path.join(input_dir, "IMDB TMDB Movie Metadata Big Dataset (1M).csv"),
        header=True,
        inferSchema=True,
        quote='"',  # handle quoted strings
        escape='"',  # handle quotes inside fields
        multiLine=True,  # allow multi-line fields
        ignoreLeadingWhiteSpace=True,
        ignoreTrailingWhiteSpace=True,
    )

    movie_df = movie_df.select(
        "id",
        "title",
        "revenue",
        "budget",
        "overview",
        "poster_path",
        "production_companies",
        "release_year",
        "Director",
        "Star1",
        "Star2",
        "Star3",
        "genres_list",
        "all_combined_keywords",
    )
    # casting from double to int
    movie_df = movie_df.withColumn(
        "release_year", F.col("release_year").cast("integer")
    )
    # Removing duplicates, null and empty
    movie_df = movie_df.dropDuplicates(["id"])
    movie_df = movie_df.na.drop(
        subset=[
            "title",
            "release_year",
            "overview",
            "all_combined_keywords",
            "poster_path",
        ]
    )
    movie_df = movie_df.filter(~(movie_df["all_combined_keywords"] == "[]"))
    # Replacing
    movie_df = movie_df.na.fill(
        "a", ["production_companies", "Star1", "Star2", "Star3"]
    )
    logger.info("Stage 1: Cleaned data saved")
    return movie_df


def combine_all_feature_columns(logger, movie_df):
    """Converting the list and string to pyspark array then combining all the columns into one"""
    logger.info("Combining all the feature column into one")
    # Step: Parse array-like strings into ArrayType
    array_schema = T.ArrayType(T.StringType())
    movie_df = movie_df.withColumn(
        "genres_list", F.from_json("genres_list", array_schema)
    )
    movie_df = movie_df.withColumn(
        "all_combined_keywords", F.from_json("all_combined_keywords", array_schema)
    )
    # After transforming the all_combined_keywords some null value are detected
    movie_df = movie_df.na.drop(subset=["all_combined_keywords"])

    # Converting the string to pyspark array
    movie_df = movie_df.withColumn("Director", F.split(F.col("Director"), ","))
    movie_df = movie_df.withColumn(
        "production_companies", F.split(F.col("production_companies"), ",")
    )
    movie_df = movie_df.withColumn("overview", F.split(F.col("overview"), ","))
    # Convert Star1, Star2, Star3 to arrays
    movie_df = (
        movie_df.withColumn("Star1", F.array(F.col("Star1")))
        .withColumn("Star2", F.array(F.col("Star2")))
        .withColumn("Star3", F.array(F.col("Star3")))
    )

    movie_df = movie_df.withColumn(
        "crews",
        F.concat(
            F.col("Star1"),
            F.col("Star2"),
            F.col("Star3"),
            F.col("Director"),
            F.col("production_companies"),
        ),
    )

    movie_df = movie_df.drop(
        "Star1", "Star2", "Star3", "Director", "production_companies"
    )

    # Removing the space in between the word
    movie_df = movie_df.withColumn(
        "crews", F.transform(F.col("crews"), lambda x: F.regexp_replace(x, "\\s+", ""))
    )
    movie_df = movie_df.withColumn(
        "all_combined_keywords",
        F.transform(
            F.col("all_combined_keywords"), lambda x: F.regexp_replace(x, "\\s+", "")
        ),
    )
    movie_df = movie_df.withColumn(
        "genres_list",
        F.transform(F.col("genres_list"), lambda x: F.regexp_replace(x, "\\s+", "")),
    )
    # Concating the four feature to make tags column
    movie_df = movie_df.withColumn(
        "tags",
        F.concat(
            F.col("all_combined_keywords"),
            F.col("genres_list"),
            F.col("overview"),
            F.col("crews"),
        ),
    )
    movie_df = movie_df.drop("all_combined_keywords", "overview", "crews")
    logger.info("Combining success")
    return movie_df


def nlp_preprocessing(logger, movie_df):
    """Converting the tags to string then nlp pipeline using spark-nlp"""
    logger.info("nlp preprocessing the tags column")
    # spark = sparknlp.start()
    # Step 1: Join tags into a single string
    movie_df = movie_df.withColumn("tags_str", F.concat_ws(" ", F.col("tags")))
    movie_df = movie_df.filter(F.trim(F.col("tags_str")) != "")

    # Step 2: NLP Pipeline
    document = DocumentAssembler().setInputCol("tags_str").setOutputCol("document")

    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")

    # Normalize: lowercase, remove punctuation, keep words & numbers only
    normalizer = (
        NLPNormalizer()
        .setInputCols(["token"])
        .setOutputCol("normalized")
        .setLowercase(True)
        .setCleanupPatterns(["[^a-zA-Z0-9]"])
    )  # keep only letters & numbers

    # Lemmatizer
    lemmatizer = (
        LemmatizerModel.pretrained("lemma_antbnc")
        .setInputCols(["normalized"])
        .setOutputCol("lemma")
    )

    # Stopwords removal
    stopwords_cleaner = (
        StopWordsCleaner()
        .setInputCols(["lemma"])
        .setOutputCol("clean_lemma")
        .setCaseSensitive(False)
    )

    # Final array output
    finisher = (
        Finisher()
        .setInputCols(["clean_lemma"])
        .setOutputCols(["tags_lemmatized"])
        .setOutputAsArray(True)
    )

    # Step 3: Build pipeline
    pipeline = Pipeline(
        stages=[
            document,
            tokenizer,
            normalizer,
            lemmatizer,
            stopwords_cleaner,
            finisher,
        ]
    )

    # Step 4: Fit + Transform
    model = pipeline.fit(movie_df)
    movie_df = model.transform(movie_df)
    logger.info("nlp preprocessing success")
    return movie_df


def vectorizing_the_word(logger, movie_df):
    """vectorizing the word then normalizing the vector"""
    logger.info("Norm-vectorizing the word")
    # vectorizing the word
    cv = CountVectorizer(inputCol="tags_lemmatized", outputCol="features")
    cv_model = cv.fit(movie_df)
    movie_df = cv_model.transform(movie_df)

    # Normalize the vector
    normalizer = VectorNormalizer(inputCol="features", outputCol="norm_features", p=2.0)
    movie_df = normalizer.transform(movie_df)
    movie_df = movie_df.drop("tags", "tags_str", "tags_lemmatized", "features")
    logger.info("Norm-vectorizing the word success")
    return movie_df


def train_lsh_model(logger, movie_df):
    """Train the BucketedRandomProjectionLSH"""
    logger.info("Training the model")
    lsh = BucketedRandomProjectionLSH(
        inputCol="norm_features",
        outputCol="hashes",
        bucketLength=2,  # tune
        numHashTables=10,  # tune
    )
    lsh_model = lsh.fit(movie_df)
    return lsh_model


def save_data_in_parquet(logger, movie_df, lsh_model, output_dir):
    """Saving the dataframe and model as parquet"""
    logger.info("Saving the movie_metadata into parquet")
    movie_metadata = movie_df.select("id", "title", "poster_path", "release_year")
    movie_metadata.write.mode("overwrite").parquet(
        os.path.join(output_dir, "stage1", "movie_metadata")
    )
    logger.info("Saved the movie_metadata into parquet")

    logger.info("Saving the lsh_model into parquet")
    lsh_model.write().overwrite().save(os.path.join(output_dir, "stage2", "lsh_model"))
    logger.info("Saved the lsh_model into parquet")

    logger.info("Saving the master into parquet")
    master = movie_df.select(
        "id", "title", "poster_path", "revenue", "budget", "release_year", "genres_list"
    )
    master.write.mode("overwrite").parquet(
        os.path.join(output_dir, "stage3", "master_table")
    )
    logger.info("Saved the master into parquet")

    logger.info("Saving the vector into parquet")
    vector = movie_df.select("id", "norm_features")
    vector.write.mode("overwrite").parquet(os.path.join(output_dir, "stage4", "vector"))
    logger.info("Saved the vector into parquet")


if __name__ == "__main__":

    logger = setup_logging("transform.log")

    if len(sys.argv) != 8:
        logger.critical(
            "Usage: python3 transform/execute.py <input_dir> <output_dir> master_ip d_mem e_mem e_core e_inst"
        )
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    spark_config = {}
    spark_config["master_ip"] = sys.argv[3]
    spark_config["driver_memory"] = sys.argv[4]
    spark_config["executor_memory"] = sys.argv[5]
    spark_config["executor_cores"] = sys.argv[6]
    spark_config["executor_instances"] = sys.argv[7]

    start = time.time()
    spark = create_spark_session(logger, spark_config)
    movie_df = load_and_clean(logger, spark, input_dir)
    movie_df = combine_all_feature_columns(logger, movie_df)
    movie_df = nlp_preprocessing(logger, movie_df)
    movie_df = vectorizing_the_word(logger, movie_df)
    lsh_model = train_lsh_model(logger, movie_df)
    save_data_in_parquet(logger, movie_df, lsh_model, output_dir)

    end = time.time()
    logger.info("Transformation pipeline completed")
    logger.info(f"Total time taken {format_time(end-start)}")
