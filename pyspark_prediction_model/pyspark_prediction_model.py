# Import libraries
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
my_spark = SparkSession.builder.appName("SalesForecast").getOrCreate()

# Importing dataset
sales_data = my_spark.read.csv("Online Retail.csv", header=True, inferSchema=True, sep=",")

# Checking dataset details
sales_data.printSchema()
print('Num records: ', sales_data.count())

# Convert InvoiceDate to datetime
sales_data = sales_data.withColumn("InvoiceDate", to_date(to_timestamp(col("InvoiceDate"), "d/M/yyyy H:mm")))

# Aggregate data into daily rows
daily_sales_data = sales_data.groupBy("Country", "StockCode", "InvoiceDate", "Year", "Month", "Day", "Week",
                                      "DayOfWeek").agg({"Quantity": "sum", "UnitPrice": "avg"})

# Rename the target column for consistency
daily_sales_data = daily_sales_data.withColumnRenamed("sum(Quantity)", "Quantity")

# Split the data into two sets based on the splitting date, "2011-09-25"
split_date_train_test = "2011-09-25"

# Creating the train and test datasets
train_data = daily_sales_data.filter(col("InvoiceDate") <= split_date_train_test)
test_data = daily_sales_data.filter(col("InvoiceDate") > split_date_train_test)

# Creating indexer for categorical columns
country_indexer = StringIndexer(inputCol="Country", outputCol="CountryIndex").setHandleInvalid("keep")
stock_code_indexer = StringIndexer(inputCol="StockCode", outputCol="StockCodeIndex").setHandleInvalid("keep")

# Selecting features columns
feature_cols = ["CountryIndex", "StockCodeIndex", "Month", "Year", "DayOfWeek", "Day", "Week"]

# Using vector assembler to combine features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Initializing a Random Forest model
rf = RandomForestRegressor(featuresCol="features", labelCol="Quantity", maxBins=4000)

# Create a pipeline for staging the processes
pipeline = Pipeline(stages=[country_indexer, stock_code_indexer, assembler, rf])

# Training the model
model = pipeline.fit(train_data)

# Getting test predictions
test_predictions = model.transform(test_data)
test_predictions = test_predictions.withColumn("prediction", col("prediction").cast("double"))

# Providing the Mean Absolute Error (MAE) for your forecast, returning a double(float) "mae"

# Initializing the evaluator
mae_evaluator = RegressionEvaluator(labelCol="Quantity", predictionCol="prediction", metricName="mae")

# Obtaining MAE
mae = mae_evaluator.evaluate(test_predictions)
print('MAE: ', mae)

# Finding how many units are sold during the  week 39 of 2011 as an integer `quantity_sold_w39`.

# Getting the weekly sales
weekly_test_predictions = test_predictions.groupBy("Year", "Week").agg({"prediction": "sum"})

# Finding the quantity sold on the 39 week.
week39 = weekly_test_predictions.filter(col('Week') == 39)

# Storing prediction as quantity_sold_w30
quantity_sold_w39 = int(week39.select("sum(prediction)").collect()[0][0])
print('Quantity sold on week 39: ', quantity_sold_w39)

# Stopping Spark session
my_spark.stop()
