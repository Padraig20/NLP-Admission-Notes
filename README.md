# NLP_admissionNotes
The goal of the project is to train an NLP model which can extract diagnoses and diseases from admission notes. The dataset is publicly available here: http://trec-cds.org/topics2021.xml

## Explanation

./datasets contains all the data I used for testing and training.

./datasets/origin contains all raw data. It is publicly available in http://trec-cds.org/topics2021.xml

./datasets/testing contains all preprocessed data I used for testing the models.

./datasets/training contains all preprocessed data I used for training the models.


./scripts contains all scripts (in Python) which I use for data processing, analysis, training and using the pretrained models.

./scripts/text_extractor.py is used for converting the original .xml file into a csv file with columns: text_id, text

./scripts/entity_extractor.py is used for knitting together two .csv files into one, which can then be used for .conll file preparation.

./scripts/conll_prep.py is used for knitting together two .csv files, one with annotated entities and one generated from text_extractor.py

./scripts/nlp_apply.py takes a pretrained model and applies it to the input text. Represents the result in readable manner.

./scripts/training/nlp_train.py generates an nlp model and automatically tests it. Saves the model to specified folder. Test and training data must be in .conll format.

./scripts/training/*.conll are used for training and testing respectively. These files have been generated from conll_prep.py

./scripts/training/ner_graphs is used for storing the graphs generated during training the model.

./scripts/training/ner_logs is used for storing the logs generated during training the model.


Here are some great tutorials I used:

https://www.johnsnowlabs.com/the-ultimate-guide-to-building-your-own-ner-model-with-python/

https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.3.prepare_CoNLL_from_annotations_for_NER.ipynb

## Dataset Explanation

The dataset contains 75 pseudonymized admission notes. Admission notes always have a similar structure, containing personal data and diagnoses of a single patient. Our goal is to extract these diagnoses and diseases with an NLP model.

Example for an admission note:


"Patient is a 45-year-old man with a history of anaplastic astrocytoma of the spine complicated by severe lower extremity weakness and urinary retention s/p Foley catheter, high-dose steroids, hypertension, and chronic pain. The tumor is located in the T-L spine, unresectable anaplastic astrocytoma s/p radiation. Complicated by progressive lower extremity weakness and urinary retention. Patient initially presented with RLE weakness where his right knee gave out with difficulty walking and right anterior thigh numbness. MRI showed a spinal cord conus mass which was biopsied and found to be anaplastic astrocytoma. Therapy included field radiation t10-l1 followed by 11 cycles of temozolomide 7 days on and 7 days off. This was followed by CPT-11 Weekly x4 with Avastin Q2 weeks/ 2 weeks rest and repeat cycle."

## Approach

### Goals in data preparation

For training the nlp model, we require two .csv files: one containing the text, and one containing the entities, along with all kinds of metadata.

The text file has following columns: text_id, text

The entities file has following columns: text_id, entity, begin, end, chunk

### First step: Converting the .xml file

For that, I used the script ./scripts/text_extractor.py - nothing interesting going on here.

### Second step: Extracting the entities

I tried doing that manually, but that was way too much work for my liking - especially extracting begin and end of each word. I gave up on that after the first text!

So I tried a different approach. I just wrote each entity I found into a custom .csv file and connected them to their text_id. This means that I got a handcrafted .csv file with the columns: text_id, entity

Then we use the ./scripts/entity_extractor.py script for filling out the rest!

How do we deal with overlaps, though? For instance, we might have "diabetes mellitus" and "diabetes" in the same word.

Interestingly, the NLP model training can handle that just fine - it even yields greater results!

### Third step: Data Splitting

Since we only have a small dataset (75 entries), I used the first 70 entries for training, and the last 5 entries for testing.

### Fourth step: .conll file preparation

For that, I used the ./scripts/conll_prep.py script, inspired from https://github.com/JohnSnowLabs/spark-nlp-workshop/blob/master/tutorials/Certification_Trainings/Healthcare/1.3.prepare_CoNLL_from_annotations_for_NER.ipynb

### First Training Session

The first training session took place with 7 epochs, a learning rate of 0.004, a batch size of 32, and a validation split of 20%. It already produced great results, despite no hyperparameter optimization. It yields an f1-score of about 60%. Results may look like:
                                                     
|   entity|      token|
|-----:|-----------|
|        O|    Patient|
|        O|         is|
|        O|       80yo|
|        O|          ,|
|        O|        has|
|        O|       been|
|        O|   recently|
|        O|  diagnosed|
|        O|       with|
|B-DISEASE|       type|
|I-DISEASE|          2|
|I-DISEASE|   diabetes|
|I-DISEASE|   mellitus|
|        O|          .|
|        O|    Suffers|
|        O|       from|
|B-DISEASE|proteinuria|
|        O|        and|
|B-DISEASE|      liver|
|I-DISEASE|  cirrhosis|


B-DISEASE -> beginning of disease entity

I-DISEASE -> inside of disease entity

O -> no entity

### Hyperparameter Tuning and Optimization

Turns out, this is a little more tricky with spark-nlp than previously thought. Although the NerDLApproach class offerst automatic percentile folds for cross validation
(with f1-score, recall etc being logged in every epoch!!), there is no way to automatically perform hyperparameter tuning with this method. We cannot even extract the f1-score.
That's kinda disappointing :(

I thought that the CrossValidator-class would be of help for that. However, it doesn't work with our predefined pipeline! It seems to not be built for NER-models after all... 
Which is unfortunate, and I think that the spark-nlp developer team should take this into consideration for future development. We need both embeddings and graph_builder before
our nerTagger. However, as soon as we use the CrossValidator, the vocabulary is not yet ready for validation. 

We are in need for a different approach!!

When testing with out test-dataset, we have both ground_truth and prediction columns. We can use the "positive-positive"'s, the "positive-negative"'s, and so on, for manually
calculating the f1-score: https://towardsdatascience.com/the-f1-score-bec2bbc38aa6

But we got a function for that! Saves us at least some work.

Normally, we would split the original dataset into three subsets: training, validation (for hyperparameter tuning) and testing (for evaluation of the final model). Our dataset
is really small, however. We need the data for training, since diagnoses are of rather subjective nature. We don't even have a subset for validation, so we are really prone to
overfitting our model. Rather than going with the holdout approach, cross-validation would be much more suitable, altough more complicated to implement from scratch.

But that's exactly what we're doing, using Pandas and KFold from the sk-learn package: https://towardsdatascience.com/effortless-hyperparameters-tuning-with-apache-spark-20ff93019ef2

Now we need to define a search space for our hyperparameters. From experience, I found the following helpful: 

learning rate: [0.005, 0.01, 0.1], hidden layers: [10, 20], batch size: [16, 32], maximum epochs: [5, 7, 10]

Finally, we iterate through the search space via the grid search approach - "fuck around and find out"

Run the script ./scripts/training/hyperparameter_tuning.py, grab a coffee, take a walk, maybe even start a family... it's gonna take forever to finish.

### Future

-> Stemming
