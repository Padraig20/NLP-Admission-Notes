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

graph_folder = "./ner_graphs"

def calculate_f1_score(ground_truth, predictions):
    from sklearn.metrics import precision_recall_fscore_support
    # Calculate precision, recall, and F1 score
    precision, recall, f1_score, _ = precision_recall_fscore_support(ground_truth, predictions, average='weighted')
    return f1_score # judge performance according to f1-score

def pipeline_tuning(lr, hiddenLayers, batchSize, maxEpochs):

    from sparknlp.annotator import TFNerDLGraphBuilder

    graph_builder = TFNerDLGraphBuilder()\
                  .setInputCols(["sentence", "token", "embeddings"])\
                  .setLabelColumn("label")\
                  .setGraphFile("auto")\
                  .setGraphFolder(graph_folder)\
                  .setHiddenUnitsNumber(hiddenLayers)

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
                  .setMaxEpochs(maxEpochs)\
                  .setLr(lr)\
                  .setBatchSize(batchSize)\
                  .setRandomSeed(0)\
                  .setIncludeConfidence(True)\
                  .setGraphFolder(graph_folder)\
                  .setOutputLogsPath('ner_logs') 

    # Define the pipeline            
    ner_pipeline = Pipeline(stages=[embeddings,
                                    graph_builder,
                                    nerTagger])

    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_results = []

    training_data_pd = training_data.toPandas()
    schema = training_data.schema

    for train_index, val_index in kf.split(training_data_pd):
        split_train_data_pd = training_data_pd.iloc[train_index]
        split_val_data_pd = training_data_pd.iloc[val_index]

        split_train_data = spark.createDataFrame(split_train_data_pd, schema)
        split_val_data = spark.createDataFrame(split_val_data_pd, schema)

        model = ner_pipeline.fit(split_train_data)

        predictions = model.transform(split_val_data)

        tb_analyzed = predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                                  predictions.label.result,
                                                  predictions.ner.result)).alias("cols"))\
                    .select(F.expr("cols['1']").alias("ground_truth"),
                            F.expr("cols['2']").alias("prediction"))

        from sklearn.metrics import precision_recall_fscore_support

        # Convert Spark DataFrame to Pandas DataFrame
        ground_truth_df = tb_analyzed.select("ground_truth").toPandas()
        predictions_df = tb_analyzed.select("prediction").toPandas()

        # Extract the required columns as arrays
        ground_truth = ground_truth_df['ground_truth'].values.tolist()
        predictions = predictions_df['prediction'].values.tolist()

        f1_score = calculate_f1_score(ground_truth, predictions)

        cv_results.append(f1_score)

    avg_f1_score = sum(cv_results) / 5

    print("---------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------")
    print("f1-score of ", avg_f1_score)
    print("Learning Rate: ", lr)
    print("Hidden Layers: ", hiddenLayers)
    print("Batch Size: ", batchSize)
    print("Maximum Epochs: ", maxEpochs)
    print("---------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------")

    return avg_f1_score

maxScore = 0
bestLr = 0
bestHiddenLayers = 0
bestBatchSize = 0
bestMaxEpochs = 0

# pre-defined search space
for lr in [0.005, 0.01, 0.1]:
    for hiddenLayers in [10, 20]:
        for batchSize in [16, 32]:
            for maxEpochs in [5, 7, 10]:
                f1_score = pipeline_tuning(lr, hiddenLayers, batchSize, maxEpochs)
                if maxScore < f1_score:
                    maxScore = f1_score
                    bestLr = lr
                    bestHiddenLayers = hiddenLayers
                    bestBatchSize = batchSize
                    bestMaxEpochs = maxEpochs

print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("f1-score of ", maxScore)
print("Learning Rate: ", bestLr)
print("Hidden Layers: ", bestHiddenLayers)
print("Batch Size: ", bestBatchSize)
print("Maximum Epochs: ", bestMaxEpochs)
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")

## TESTING THE MODEL

test_data = CoNLL().readDataset(spark, testing_path)

predictions = ner_model.transform(test_data)

tb_analyzed = predictions.select(F.explode(F.arrays_zip(predictions.token.result,
                                          predictions.label.result,
                                          predictions.ner.result)).alias("cols"))\
            .select(F.expr("cols['1']").alias("ground_truth"),
                    F.expr("cols['2']").alias("prediction"))

# Convert Spark DataFrame to Pandas DataFrame
ground_truth_df = tb_analyzed.select("ground_truth").toPandas()
predictions_df = tb_analyzed.select("prediction").toPandas()

# Extract the required columns as arrays
ground_truth = ground_truth_df['ground_truth'].values.tolist()
predictions = predictions_df['prediction'].values.tolist()

f1_score = calculate_f1_score(ground_truth, predictions)

print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("Got f1-score from testing data of ", f1_score)
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")
print("---------------------------------------------------------------------------------------------")