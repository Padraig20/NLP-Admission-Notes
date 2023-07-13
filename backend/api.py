from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.ml import PipelineModel
from sparknlp.training import CoNLL
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import sparknlp
import sys

def handle_request(text):
    data = spark.createDataFrame([[text]]).toDF("text")

    tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token").fit(data)
    pipeline = Pipeline().setStages([documentAssembler, sentence, tokenizer]).fit(data)

    import pyspark.sql.functions as F
    #data = data.withColumn("text", F.lower(data["text"]))

    tokenized = pipeline.transform(data)

    inputData = tokenized.drop("text")

    result = loaded_model.transform(inputData)

    ##DISPLAY RESULTS

    entities = result.select("ner.result").rdd.flatMap(lambda x: x).collect()
    tokens = result.select("token.result").rdd.flatMap(lambda x: x).collect()

    return tokens, entities


if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    print("Usage: python api.py model_path")
    sys.exit(1)

app = Flask(__name__)
CORS(app)  # Initialize CORS
spark = sparknlp.start()

loaded_model = PipelineModel.load(model_path)

documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
sentence = SentenceDetector()\
    .setInputCols(["document"])\
    .setOutputCol("sentence")
print("Serving API now...")

@app.route('/extract_entities', methods=['POST'])
def main():
    text = request.get_data(as_text=True)
    result = handle_request(text)
    return jsonify({'tokens': result[0], 'entities': result[1]})

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer # Imports the WSGIServer
    from gevent import monkey; monkey.patch_all()
    LISTEN = ('0.0.0.0',8080)
    http_server = WSGIServer( LISTEN, app )
    http_server.serve_forever()
