from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum, avg

# Step 1: Create Spark Session
spark = SparkSession.builder \
    .appName("Sales Data Pipeline") \
    .getOrCreate()

# Step 2: Load Data
df = spark.read.csv("data/sales_data.csv", header=True, inferSchema=True)

print("Initial Data:")
df.show()

# Step 3: Data Cleaning
df_clean = df.dropna()

# Step 4: Add Total Amount Column
df_transformed = df_clean.withColumn(
    "total_amount", col("price") * col("quantity")
)

# Step 5: Aggregations
category_sales = df_transformed.groupBy("category").agg(
    sum("total_amount").alias("total_sales"),
    avg("total_amount").alias("avg_sales")
)

print("Category-wise Sales:")
category_sales.show()

# Step 6: Filter High Sales
high_sales = df_transformed.filter(col("total_amount") > 10000)

print("High Value Transactions:")
high_sales.show()

# Step 7: Save Output
category_sales.write.mode("overwrite").csv("output/category_sales", header=True)

print("Pipeline executed successfully!")