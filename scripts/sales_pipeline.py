from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Missing').getOrCreate()
training = spark.read.csv('D:/Excel Files/Employee data.csv',header = True,inferSchema= True)
training.show()
training.columns

from pyspark.ml.feature import Imputer
imputer = Imputer(
    inputCols=['age','experience','salary'],
    outputCols=['age','experience','salary']
)
df = imputer.fit(training).transform(training)
#Grouping the independent features
from pyspark.ml.feature import VectorAssembler
featureassembler = VectorAssembler(inputCols=["age","experience"],outputCol="Independent Features")
output = featureassembler.transform(df)
output.show()

finalised_data = output.select(['Independent features','salary'])
finalised_data.show()

#Now performing the same machine learning thing(train test split)
from pyspark.ml.regression import LinearRegression
train_data,test_data= finalised_data.randomSplit([0.75,0.25])
regressor = LinearRegression(featuresCol='Independent features',labelCol='salary')
regressor = regressor.fit(train_data)

regressor.coefficients
regressor.intercept

pred_results= regressor.evaluate(test_data)
pred_results.predictions.show()

pred_results.meanAbsoluteError,pred_results.meanSquaredError
