#!/usr/bin/python

# Package imports and configuration parameters
print("Package imports and configuration parameters")

import matplotlib.pyplot as plt

from PyQt5 import QtWidgets
qApp = QtWidgets.QApplication([str(' ')])

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

conf = SparkConf().setAppName("QualitativeBinomialModel").setMaster('local[*]')
context = SparkContext(conf=conf)
context.setLogLevel('ERROR')
session = SparkSession(context)

import warnings
warnings.filterwarnings(action='ignore')



# Load data into a RDD
print("\nLoad data into a RDD")

from pyspark import SparkFiles

train_file = "./DATA/kddcup.data.gz"
test_file  = "./DATA/corrected.gz"

train_data = context.textFile(train_file)
test_data  = context.textFile(test_file)
print(train_data.count())
print(test_data.count())
print(train_data.take(1))



# Splitting data in subgroups
print("\nSplitting data in subgroups")

train_data_1, train_data_2 = train_data.randomSplit([7, 3], 10)
test_data_1, test_data_2 = test_data.randomSplit([1, 9], 12)
train_data = train_data_1.union(train_data_2)
test_data = test_data_1.union(test_data_2)

print(train_data.count())
print(test_data.count())
print(test_data.take(1))



# Data preparation into [label, vector] tuples from the label field
# in the last column along with quantitative characteristics and the
# indexed and codified form of qualitative (categorical) characteristics.
print("\nData preparation into [label, vector] tuples from the label field " +
        "in the last column along with quantitative characteristics and the " +
        "indexed and codified form of qualitative (categorical) characteristics.")

train_data_df = train_data.map(lambda x: x.split(',')).toDF()
test_data_df  = test_data.map(lambda x: x.split(',')).toDF()

print(train_data_df.head())



from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer
from pyspark.ml import Pipeline
from pyspark.mllib.linalg import Vectors
from pyspark.sql.functions import udf, monotonically_increasing_id
from pyspark.sql.types import ArrayType, DoubleType, IntegerType

train_cat_cols = train_data_df.select(train_data_df.columns[1:4])
test_cat_cols  = test_data_df.select(test_data_df.columns[1:4])

train_data_df = train_data_df.withColumn('id', monotonically_increasing_id()).drop(*train_data_df.columns[1:4])
test_data_df  = test_data_df.withColumn('id', monotonically_increasing_id()).drop(*test_data_df.columns[1:4])

label_column = train_data_df.columns[-1]
label_index  = train_data_df.columns.index(label_column) - 1

stages = [ StringIndexer(inputCol='_'+str(i+1), outputCol='ind'+str(i), handleInvalid='keep') for i in range(1, 4) ]
stages.append(OneHotEncoderEstimator(inputCols=[ 'ind'+str(i) for i in range(1,4) ], outputCols=[ 'cat'+str(i) for i in range(1,4) ], dropLast=False))
pipeline = Pipeline(stages=stages)
categorization_model = pipeline.fit(train_cat_cols)

col_mapping_udf = udf(lambda c: c.toArray().tolist(), returnType=ArrayType(DoubleType()))



def vectorize_categoric_columns(df, cols):
    dataframe = categorization_model.transform(df)
    for c in cols:
        dataframe = dataframe.withColumn(c, col_mapping_udf(c))
    return dataframe

cat_cols = [ 'cat' + str(i) for i in range(1, 4) ]
train_vect_cols = vectorize_categoric_columns(train_cat_cols, cat_cols).select(cat_cols)
test_vect_cols  = vectorize_categoric_columns(test_cat_cols,  cat_cols).select(cat_cols)

example_row = train_vect_cols.take(1)[0]
num_vals_col = [ len(col) for col in example_row ]

print(example_row)
print(num_vals_col)



def split_vectorized_columns(df):
    vect_cols = []
    for i in range(len(num_vals_col)):
        lab = 'cat' + str(i + 1)
        vect_cols.append(df.select([ df[lab][j].cast(IntegerType()).alias(lab+'_'+str(j+1)) for j in range(num_vals_col[i]) ]))
    return vect_cols

train_vect_cols = split_vectorized_columns(train_vect_cols)
test_vect_cols  = split_vectorized_columns(test_vect_cols)

print(train_vect_cols)



def merge_vectorized_cols_df(vect_cols_df):
    vect_cols = [ vcdf.withColumn('id', monotonically_increasing_id()) for vcdf in vect_cols_df ]
    for i in range(1, len(num_vals_col)):
        vect_cols[0] = vect_cols[0].join(vect_cols[i], 'id', 'outer')
    return vect_cols[0]

train_vect_df = merge_vectorized_cols_df(train_vect_cols)
test_vect_df  = merge_vectorized_cols_df(test_vect_cols)

print(train_vect_df)



train_data_df = train_data_df.join(train_vect_df, 'id', 'outer').drop('id')
test_data_df  = test_data_df.join(test_vect_df, 'id', 'outer').drop('id')

print(train_data_df)



import numpy as np

def prepare_data(data):
    label = data[label_index]
    vector = np.array(data[: label_index] + data[label_index+1 :])
    return (label, vector)

tr_threat = train_data_df.rdd.map(prepare_data)
print(tr_threat.count())
print(tr_threat.take(1))

ts_threat = test_data_df.rdd.map(prepare_data)
print(ts_threat.count())
print(ts_threat.take(1))



# Descriptive statistics
print("\nDescriptive statistics")

from operator import add

ltr_threat = tr_threat.map(lambda x: x[0])
tr_threat  = tr_threat.map(lambda x: x[1])

lts_threat = ts_threat.map(lambda x: x[0])
ts_threat  = ts_threat.map(lambda x: x[1])

def descriptive_statistics(labelled_threat):
    threat_count = labelled_threat.map(lambda lt: (lt, 1)).reduceByKey(add).collect()

    threat_labels = list(map(lambda x: x[0], threat_count))
    threat_values = list(map(lambda x: x[1], threat_count))
    y_pos = np.arange(len(threat_labels))

    plt.bar(y_pos, threat_values, align='center', alpha=0.5)
    plt.xticks(y_pos, threat_labels, rotation=90)
    plt.ylabel("Absolute Frequency")
    plt.show()

    return zip(threat_labels, threat_values)

labels = [ lab[0][: -1] for lab in descriptive_statistics(ltr_threat) ]

descriptive_statistics(lts_threat)



# Scaling characteristics to mu=0 and delta=1
print("\nScaling characteristics to mu=0 and delta=1")

from pyspark.mllib.feature import StandardScaler

std = StandardScaler(withMean=True, withStd=True)
model = std.fit(tr_threat)

std_tr_threat = model.transform(tr_threat)
std_ts_threat = model.transform(ts_threat)

print("Train:", std_tr_threat.take(1))
print("Test:",  std_ts_threat.take(1))



# Preparation of binomial labels
print("\nPreparation of binomial labels")

from pyspark.mllib.regression import LabeledPoint

std_train_data = ltr_threat.zip(std_tr_threat)
std_test_data  = lts_threat.zip(std_ts_threat)

print(std_train_data.take(1))



def prepare_binary_classifier_data(label, data_vector):
    attack = 0 if label.find('normal') >= 0 else 1
    return LabeledPoint(attack, data_vector)

train_data = std_train_data.map(lambda x: prepare_binary_classifier_data(x[0], x[1]))
test_data  = std_test_data.map(lambda x:  prepare_binary_classifier_data(x[0], x[1]))

print(train_data.count())
print(train_data.take(20))



# Descriptive statistics on the labelled points of data, having been scaled and mapped.
print("\nDescriptive statistics on the labelled points of data, having been scaled and mapped.")

tr_threat  = train_data.map(lambda p: p.features)
ltr_threat = train_data.map(lambda p: p.label > 0)

ts_threat  = test_data.map(lambda p: p.features)
lts_threat = test_data.map(lambda p: p.label > 0)

descriptive_statistics(ltr_threat)
descriptive_statistics(lts_threat)



# Representation of the logistic function
print("\nRepresentation of the logistic function")

x = np.linspace(-10, 10, 1000)
plt.plot(x, 1 / (1 + np.exp(-0.5 * x)))
plt.title("Logistic function")
plt.show()



# Modelling
print("\nModelling")

from pyspark.mllib.classification import LogisticRegressionWithLBFGS, LogisticRegressionModel

logistic_model = LogisticRegressionWithLBFGS.train(train_data, numClasses=2, intercept=False)

print(logistic_model.weights)



# Evaluating the model
print("\nEvaluating the model")

label_and_prediction = test_data.map(lambda p: (p.label, logistic_model.predict(p.features)))

def count_OK_vs_KO(value):
    real_value = value[0]
    predicted_value = value[1]

    result = 'OK' if real_value == predicted_value else 'KO'
    return (result, 1)

results = label_and_prediction.map(count_OK_vs_KO).reduceByKey(add).sortByKey(ascending=False).collect()

print(results)



def compute_error(results):
    results_matrix = {}
    results_matrix[results[0][0]] = results[0][1]
    results_matrix[results[1][0]] = results[1][1]

    error = float(results_matrix['KO']) * 100 / (float(results_matrix['OK']) + float(results_matrix['KO']))
    return error

error = compute_error(results)
print("Error: {}%".format(error, 2))



from shutil import rmtree

path = "./QualitativeBinomialModel"

try:
    rmtree(path)
except:
    pass

logistic_model.save(context, path)



# Production-ready model
print("\nProduction-ready model")

logistic_model_prod = LogisticRegressionModel.load(context, path)



# Confusion matrix from the binary classifier
print("\nConfusion matrix from the binary classifier")

def count_confusion(values):
    real_value = values[0]
    predicted_value = values[1]
    return (str(int(real_value)) + '_' + str(int(predicted_value)), 1)

classification_result = label_and_prediction.map(count_confusion).reduceByKey(add).collect()
classification_result = np.sort(np.reshape(classification_result, (len(classification_result), 2)), axis=0)
print(classification_result)



def compute_absolute_freq_matrix(classification):
    num_classes = 2
    matrix = np.zeros((num_classes, num_classes), dtype=np.uint32)
    for result in classification:
        c = [ int(r) for r in result[0].split('_') ]
        matrix[c[0], c[1]] = result[1]
    return matrix

abs_freq_matrix = compute_absolute_freq_matrix(classification_result)
print(abs_freq_matrix)
