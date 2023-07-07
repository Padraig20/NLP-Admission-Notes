import sys
import os

if len(sys.argv) != 4:
    print("Usage: python nlp_train.py training_data.conll testing_data.conll output_name")
    sys.exit(1)

training_path = sys.argv[1]
testing_path = sys.argv[2]
output_name = sys.argv[3]

if not training_path.endswith(".conll") or not testing_path.endswith(".conll"):
    print("Input files must have the extension '.conll'")
    sys.exit(1)

if '.' in output_name:
    print("Output name cannot have an extension")
    sys.exit(1)

##START SPARK-NLP LOGIC

import sparknlp

spark = sparknlp.start()

from sparknlp.training import CoNLL
training_data = CoNLL().readDataset(spark, training_path)

training_data.show(3)

import pyspark.sql.functions as F
training_data.select(F.explode(F.arrays_zip(training_data.token.result,
                                            training_data.label.result)).alias("cols")) \
             .select(F.expr("cols['0']").alias("token"),
                     F.expr("cols['1']").alias("ground_truth")).groupBy('ground_truth').count().orderBy('count', ascending=False).show(100,truncate=False)

training_data = training_data.withColumn("text", F.lower(training_data["text"]))

graph_folder = "./ner_graphs"

from sparknlp.annotator import TFNerDLGraphBuilder

graph_builder = TFNerDLGraphBuilder()\
              .setInputCols(["sentence", "token", "embeddings"])\
              .setLabelColumn("label")\
              .setGraphFile("auto")\
              .setGraphFolder(graph_folder)\
              .setHiddenUnitsNumber(10)

from sparknlp.base import DocumentAssembler, Pipeline
from sparknlp.annotator import (
    Tokenizer,
    SentenceDetector,
    BertEmbeddings
)

# Step 1: Transforms raw texts to `document` annotation
documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
# Step 2: Getting the sentences
sentence = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
# Step 3: Tokenization
tokenizer = Tokenizer()\
    .setInputCols(["sentence"])\
    .setOutputCol("token")
# Step 4: Bert Embeddings
embeddings = BertEmbeddings.pretrained().\
    setInputCols(["sentence", 'token']).\
    setOutputCol("embeddings")

from sparknlp.annotator import NerDLApproach
# Model training
nerTagger = NerDLApproach()\
              .setInputCols(["sentence", "token", "embeddings"])\
              .setLabelColumn("label")\
              .setUseBestModel(True)\
              .setOutputCol("ner")\
              .setMaxEpochs(10)\
              .setLr(0.003)\
              .setBatchSize(8)\
              .setRandomSeed(0)\
              .setVerbose(1)\
              .setValidationSplit(0.2)\
              .setEvaluationLogExtended(True)\
              .setEnableOutputLogs(True)\
              .setIncludeConfidence(True)\
              .setGraphFolder(graph_folder)\
              .setOutputLogsPath('ner_logs') 

# Define the pipeline            
ner_pipeline = Pipeline(stages=[embeddings,
                                graph_builder,
                                nerTagger])

ner_model = ner_pipeline.fit(training_data)

ner_model.save(output_name)

## TESTING THE MODEL

test_data = CoNLL().readDataset(spark, testing_path)

test_data = test_data.withColumn("text", F.lower(test_data["text"]))

predictions = ner_model.transform(test_data)

predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                          predictions.label.result,
                                          predictions.ner.result)).alias("cols"))\
            .select(F.expr("cols['0']").alias("token"),
                    F.expr("cols['1']").alias("ground_truth"),
                    F.expr("cols['2']").alias("prediction")).show(5000, truncate=False)
