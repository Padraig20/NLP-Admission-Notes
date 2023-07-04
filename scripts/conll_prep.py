import json

import pandas as pd
from tqdm import tqdm
from collections import Counter

import sparknlp
from sparknlp.base import *
from sparknlp.common import *
from sparknlp.annotator import *
from sparknlp.training import CoNLL

import pyspark.sql.functions as F
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

import sys
import os

# Check if the correct number of command-line arguments is provided
if len(sys.argv) != 4:
    print("Usage: python conll_prep.py text.csv entities.csv output_file.conll")
    sys.exit(1)

# Extract the filenames from the command-line arguments
text_path = sys.argv[1]
entities_path = sys.argv[2]
output_file = sys.argv[3]

# Check if the input file extensions are ".csv"
if not input_file1.endswith(".csv") or not input_file2.endswith(".csv"):
    print("Input files must have the extension '.csv'")
    sys.exit(1)

# Check if the output file extension is ".conll"
if not output_file.endswith(".conll"):
    print("Output file must have the extension '.conll'")
    sys.exit(1)


spark = sparknlp.start()

print ("Spark NLP Version :", sparknlp.version())

train_text_df = pd.read_csv(text_path)
print(train_text_df.head())

train_entities_df = pd.read_csv(entities_path)
train_entities_df= train_entities_df[["text_id", "begin", "end", "chunk", "entity"]]
print(train_entities_df.head())


def make_conll(text:pd.DataFrame, entity:pd.DataFrame,
               save_tag:bool=None,
               save_conll:bool=None,
               verbose:bool=None,
               begin_deviation:int=0,
               end_deviation:int=0 )->str:

    df_text = text.iloc[:,[0,1]]
    df_entity = entity.iloc[:,[0,1,2,3,4]]
    df_text.columns = ['text_id','text']
    df_entity.columns = ['text_id','begin','end','chunk','entity']
    entity_list = list(df_entity.entity.unique())


    ########--------------1.tag transformation function------------########

    def transform_text(text, entities, verbose=None):

        tag_list=[]
        for entity in entities.iterrows():

            begin = entity[1][1] + begin_deviation
            end = entity[1][2] + end_deviation
            chunk = entity[1][3]
            tag = entity[1][4]
            text = text[:end] + f' </END_NER:{tag}> ' + text[end:]
            text = text[:begin] + f' <START_NER:{tag}> ' + text[begin:]
            tag_list.append(tag)

        sum_of_added_entity = Counter(tag_list)
        sum_of_entity = Counter(entities['entity'].values)

        if verbose:
            print(f'Processed text id   : {entities.text_id.values[:1]}')
            print(f'Original Entities   : {sum_of_entity}\nAdded Entities      : {sum_of_added_entity}')
            print(f'Number Equality     : {sum_of_added_entity == sum_of_entity}')
            print("=="*40)

        if not sum_of_entity == sum_of_added_entity:
            print("There is a problem in text id:")
            print(entities.text_id.values[0])
            raise Exception("Check this text!")

        return text


    ######---------------2.apply_transform_text function ----------------#######

    def apply_tag_ner(df_text, df_entity, save=None, verbose=None):

        for text_id in tqdm(df_text.text_id):
            text  = df_text.loc[df_text['text_id']==text_id]['text'].values[0]
            entities  = df_entity.loc[(df_entity['text_id']==text_id)].sort_values(by='begin',ascending=False)

            df_text.loc[df_text['text_id']==text_id, 'text'] = transform_text(text, entities, verbose=verbose)

        if save:
            df_text.to_csv("text_with_ner_tag.csv", index=False, encoding='utf8')

        return df_text


    ##########----------------3.RUNNING TAG FUNCTION---------------#############

    print("Text tagging starting. Applying entities to whole text...\n")
    df = apply_tag_ner(df_text, df_entity, save=save_tag, verbose=verbose)


    ###########---------------4.Spark Pipeline-----------------------###########

    def spark_pipeline(df):
        spark_df = spark.createDataFrame(df)

        documentAssembler = DocumentAssembler()\
            .setInputCol("text")\
            .setOutputCol("document")\
            .setCleanupMode("shrink")

        sentenceDetector = SentenceDetector()\
            .setInputCols(['document'])\
            .setOutputCol('sentences')\
            .setExplodeSentences(True)

        tokenizer = Tokenizer() \
            .setInputCols(["sentences"]) \
            .setOutputCol("token")

        nlpPipeline = Pipeline(stages=[documentAssembler, sentenceDetector, tokenizer ])

        empty_df = spark.createDataFrame([['']]).toDF("text")
        pipelineModel = nlpPipeline.fit(empty_df)

        result = pipelineModel.transform(spark_df.select(['text']))


        return result.select('token.result').toPandas()
    print("\n\nSpark pipeline is running...")
    df_final = spark_pipeline(df)


    #########--------------5.CoNLL Function--------------------#############

    def build_conll(df_final, tag_list, save=None):

        header = "-DOCSTART- -X- -X- O\n\n"
        conll_text = ""
        chunks = []
        tag_list = tag_list
        tag = 'O'      # token tag
        ct = 'B'       # chunk tag part B or I

        for sentence_tokens in tqdm(df_final.result[:]):
            for token in sentence_tokens:
                if token.startswith("<START_NER:"):
                    tag = token.split(':')[1][:-1]
                    if tag not in tag_list:
                        tag = 'O'
                        conll_text += f'{token} NN NN {tag}\n'

                    continue

                if token.startswith("</END_NER:") and tag != 'O':
                    for i, chunk in enumerate(chunks):
                        ct = 'B' if i == 0 else 'I'
                        conll_text += f'{chunk} NNP NNP {ct}-{tag}\n'

                    chunks=[]
                    tag='O'
                    continue

                if tag != 'O':
                    chunks.append(token)
                    continue

                if tag == 'O':
                    conll_text += f'{token} NN NN {tag}\n'
                    continue

            conll_text += '\n'

        print("\nDONE!")
        return conll_text


    ########----------------6.RUNNING CONLL FUNCTION--------------------########

    print("Conll file is being created...\n")
    return build_conll(df_final, tag_list=entity_list, save=save_conll)

# if you want tagged text or conll file saved in the current directory: just make default 'save_tag' or 'save_conll' parameters True.
conll_text = make_conll(train_text_df, train_entities_df, save_conll=True)

# Checking conll string
print(conll_text[:532])

# save as file
with open(output_file, "w+", encoding='utf8') as f:
    f.write("-DOCSTART- -X- -X- O\n\n")
    f.write(conll_text)

