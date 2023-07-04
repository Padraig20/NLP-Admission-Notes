from pyspark.ml import PipelineModel
from sparknlp.training import CoNLL
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import sparknlp
import sys

if len(sys.argv) > 1:
    string_param = sys.argv[1]
else:
    print("No input provided.")

spark = sparknlp.start()

loaded_model = PipelineModel.load("bert_diseases")

data = spark.createDataFrame([[string_param]]).toDF("text")

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentence = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token").fit(data)

pipeline = Pipeline().setStages([documentAssembler, sentence, tokenizer]).fit(data)
tokenized = pipeline.transform(data)

inputData = tokenized.drop("text")

result = loaded_model.transform(inputData)

##DISPLAY RESULTS

entities = result.select("ner.result")
tokens = result.select("token.result")

from pyspark.sql.functions import first

entities_first_row = entities.select(first("result").alias("entities"))
tokens_first_row = tokens.select(first("result").alias("tokens"))
joined_result = entities_first_row.crossJoin(tokens_first_row)

from pyspark.sql.functions import posexplode

exploded_entities = entities.select(posexplode(entities.result).alias("pos", "entity"))
exploded_tokens = tokens.select(posexplode(tokens.result).alias("pos", "token"))
joined_result = exploded_entities.join(exploded_tokens, "pos").drop("pos")

joined_result.show()
