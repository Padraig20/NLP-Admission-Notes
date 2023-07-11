from flask import Flask, request, jsonify
from flask_cors import CORS
from pyspark.ml import PipelineModel
from sparknlp.training import CoNLL
import sparknlp
from sparknlp.base import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
import sparknlp

def handle_request(text):
    loaded_model = PipelineModel.load("../scripts/training/temporary")

    data = spark.createDataFrame([[text]]).toDF("text")

    import pyspark.sql.functions as F
    #data = data.withColumn("text", F.lower(data["text"]))

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

    entities = result.select("ner.result").rdd.flatMap(lambda x: x).collect()
    tokens = result.select("token.result").rdd.flatMap(lambda x: x).collect()

    return tokens, entities

app = Flask(__name__)
CORS(app)  # Initialize CORS
spark = sparknlp.start()

@app.route('/extract_entities', methods=['POST'])
def main():
    text = request.get_data(as_text=True)
    result = handle_request(text)
    return jsonify({'tokens': result[0], 'entities': result[1]})

if __name__ == '__main__':
    from gevent.pywsgi import WSGIServer # Imports the WSGIServer
    from gevent import monkey; monkey.patch_all()
    LISTEN = ('0.0.0.0',8000)
    http_server = WSGIServer( LISTEN, app )
    http_server.serve_forever()
