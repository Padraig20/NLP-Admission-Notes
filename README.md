# NLP_admissionNotes
The objective of this project is to develop an NLP model capable of extracting diagnoses and diseases from admission notes. Admission notes are created by doctors when a patient is newly admitted to a hospital and typically contain personal information such as age, gender, and various medical conditions. However, the process of entering this information into the hospital's medical system, such as the Hospital Information System (HIS), can be time-consuming and challenging. Therefore, the aim is to train an advanced NLP model that can efficiently and accurately extract relevant diagnoses and diseases from these admission notes. The dataset containing pseudonymized admission notes is publicly available here: http://trec-cds.org/topics2021.xml

## Explanation

./hyperparameter_tuning_results.txt contains results from hyperparameter exploration.

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

./scripts/training/hyperparameter_tuning.py is used for hyperparameter optimization. Implemented Grid Search through a pre-defined search space. Cross-Validation is used for
the generated models - evaluated by f1-score. Report is automatically generated.

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

Now we need to define a search space for our hyperparameters. From experience, I found the following grid to make sense: 

learning rate: [0.001, 0.003, 0.005, 0.01, 0.1], hidden layers: [10, 20], batch size: [8, 16, 32], maximum epochs: [7, 10]

Finally, we iterate through the search space via the grid search approach - "fuck around and find out"

Run the script ./scripts/training/hyperparameter_tuning.py, grab a coffee, take a walk, maybe even start a family... it's gonna take forever to finish.

Analysis took about 3h, and once aborted due to a network error. The results can be seen in "./hyperparameter_tuning_results.txt". I was able to make a few observations during training, 
which has allowed me to skip a few extra iterations.

A bigger batch size meant less performance. A batch size of 32 meant terrible performance, which can be probably traced back to the fact that the dataset is way too small for such a batch size.
A smaller batch size often means independence from the other data in the dataset, which helps us to avoid overfitting. A batch size too small can have the adverse effect of underfitting, though.

The f1-score also got better the more epochs we had. The number of hidden layers did not necessarily have any significant effect on the performance.

Another interesting observation was that with a higher learning rate, the f1-score dropped significantly. It performed much better with lower learning rates.

The final hyperparameters is:

f1-score: 0.9506947550389957 | Learning Rate: 0.003 | Hidden Layers: 10 | Batch Size: 8 | Maximum Epochs: 10

A model trained with these parameters yields surprisingly good results with the test data - it even recognizes diagnoses which I forgot to include in data preprocessing...

|token               |ground_truth|prediction|
|-----:|-----------|-----------|
|The                 |O           |O         |
|patient             |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|34-year-old         |O           |O         |
|obese               |B-DISEASE   |B-DISEASE |
|woman               |O           |O         |
|who                 |O           |O         |
|comes               |O           |O         |
|to                  |O           |O         |
|the                 |O           |O         |
|clinic              |O           |O         |
|with                |O           |O         |
|weight              |O           |O         |
|concerns            |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|165                 |O           |O         |
|cm                  |O           |O         |
|tall                |O           |O         |
|,                   |O           |O         |
|and                 |O           |O         |
|her                 |O           |O         |
|weight              |O           |O         |
|is                  |O           |O         |
|113                 |O           |O         |
|kg                  |O           |O         |
|(                   |O           |O         |
|BMI                 |O           |O         |
|:                   |O           |O         |
|41.5                |O           |O         |
|).                  |O           |O         |
|In                  |O           |O         |
|the                 |O           |O         |
|past                |O           |O         |
|,                   |O           |O         |
|she                 |O           |O         |
|unsuccessfully      |O           |O         |
|used                |O           |O         |
|anti                |O           |O         |
|obesity             |B-DISEASE   |O         |
|agents              |O           |O         |
|and                 |O           |O         |
|appetite            |O           |O         |
|suppressants        |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|complaining         |O           |O         |
|of                  |O           |O         |
|sleep               |B-DISEASE   |B-DISEASE |
|apnea               |I-DISEASE   |I-DISEASE |
|,                   |O           |O         |
|PCO                 |B-DISEASE   |B-DISEASE |
|and                 |O           |O         |
|dissatisfaction     |O           |O         |
|with                |O           |O         |
|her                 |O           |O         |
|body                |O           |O         |
|shape               |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|high-school         |O           |O         |
|teacher             |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|married             |O           |O         |
|for                 |O           |O         |
|5                   |O           |O         |
|years               |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|doesn't             |O           |O         |
|use                 |O           |O         |
|any                 |O           |O         |
|contraceptive       |O           |O         |
|methods             |O           |O         |
|for                 |O           |O         |
|the                 |O           |O         |
|past                |O           |O         |
|4                   |O           |O         |
|months              |O           |O         |
|and                 |O           |O         |
|she                 |O           |O         |
|had                 |O           |O         |
|no                  |O           |O         |
|prior               |O           |O         |
|pregnancies         |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|rarely              |O           |O         |
|exercises           |O           |O         |
|and                 |O           |O         |
|movement            |O           |O         |
|seems               |O           |O         |
|to                  |O           |O         |
|be                  |O           |O         |
|hard                |O           |O         |
|for                 |O           |O         |
|her                 |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|not                 |O           |O         |
|able                |O           |O         |
|to                  |O           |O         |
|complete            |O           |O         |
|the                 |O           |O         |
|four-square         |O           |O         |
|step                |O           |O         |
|test                |O           |O         |
|in                  |O           |O         |
|less                |O           |O         |
|than                |O           |O         |
|15                  |O           |O         |
|seconds             |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|does                |O           |O         |
|not                 |O           |O         |
|smoke               |O           |O         |
|or                  |O           |O         |
|use                 |O           |O         |
|any                 |O           |O         |
|drugs               |O           |O         |
|.                   |O           |O         |
|Her                 |O           |O         |
|BP                  |O           |O         |
|:                   |O           |O         |
|130/80              |O           |O         |
|,                   |O           |O         |
|HR                  |O           |O         |
|:                   |O           |O         |
|195/min             |O           |O         |
|and                 |O           |O         |
|her                 |O           |O         |
|BMI                 |O           |O         |
|is                  |O           |O         |
|:                   |O           |O         |
|41.54               |O           |O         |
|.                   |O           |O         |
|Her                 |O           |O         |
|lab                 |O           |O         |
|results:FBS         |O           |O         |
|:                   |O           |O         |
|98                  |O           |O         |
|mg/dlTG             |O           |O         |
|:                   |O           |O         |
|150                 |O           |O         |
|mg/dlCholesterol    |O           |O         |
|:                   |O           |O         |
|180                 |O           |O         |
|mg/dlLDL            |O           |O         |
|:                   |O           |O         |
|90                  |O           |O         |
|mg/dlHDL            |O           |O         |
|:                   |O           |O         |
|35                  |O           |O         |
|mg/dlShe            |O           |O         |
|is                  |O           |O         |
|considering         |O           |O         |
|a                   |O           |O         |
|laparoscopic        |O           |O         |
|gastric             |O           |B-DISEASE |
|bypass              |O           |I-DISEASE |
|.                   |O           |O         |
|The                 |O           |O         |
|patient             |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|16-year-old         |O           |O         |
|girl                |O           |O         |
|recently            |O           |O         |
|diagnosed           |O           |O         |
|with                |O           |O         |
|myasthenia          |B-DISEASE   |B-DISEASE |
|gravis              |I-DISEASE   |I-DISEASE |
|,                   |O           |O         |
|class               |O           |B-DISEASE |
|IIa                 |O           |I-DISEASE |
|.                   |O           |O         |
|She                 |O           |O         |
|complains           |O           |O         |
|of                  |O           |O         |
|diplopia            |B-DISEASE   |B-DISEASE |
|and                 |O           |O         |
|weakness            |B-DISEASE   |B-DISEASE |
|affecting           |O           |O         |
|in                  |O           |O         |
|her                 |O           |O         |
|upper               |O           |O         |
|extremities         |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|had                 |O           |O         |
|a                   |O           |O         |
|positive            |O           |O         |
|anti-AChR           |O           |O         |
|antibody            |O           |O         |
|test                |O           |O         |
|,                   |O           |O         |
|and                 |O           |O         |
|her                 |O           |O         |
|single              |O           |O         |
|fiber               |O           |O         |
|electromyography    |O           |O         |
|(                   |O           |O         |
|SFEMG               |O           |O         |
|)                   |O           |O         |
|was                 |O           |O         |
|positive            |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|on                  |O           |O         |
|acetylcholinesterase|O           |O         |
|inhibitor           |O           |O         |
|treatment           |O           |O         |
|combined            |O           |O         |
|with                |O           |O         |
|immunosuppressants  |O           |O         |
|.                   |O           |O         |
|But                 |O           |O         |
|she                 |O           |O         |
|still               |O           |O         |
|has                 |O           |O         |
|some                |O           |O         |
|symptoms            |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|does                |O           |O         |
|not                 |O           |O         |
|smoke               |O           |O         |
|or                  |O           |O         |
|use                 |O           |O         |
|illicit             |O           |O         |
|drugs               |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|is                  |O           |O         |
|not                 |O           |O         |
|sexually            |O           |O         |
|active              |O           |O         |
|,                   |O           |O         |
|and                 |O           |O         |
|her                 |O           |O         |
|menses              |O           |O         |
|are                 |O           |O         |
|regular             |O           |O         |
|.                   |O           |O         |
|Her                 |O           |O         |
|physical            |O           |O         |
|exam                |O           |O         |
|and                 |O           |O         |
|lab                 |O           |O         |
|studies             |O           |O         |
|are                 |O           |O         |
|not                 |O           |O         |
|remarkable          |O           |O         |
|for                 |O           |O         |
|any                 |O           |O         |
|other               |O           |O         |
|abnormalities.BP    |O           |O         |
|:                   |O           |O         |
|110/75Hgb           |O           |O         |
|:                   |O           |O         |
|11                  |O           |O         |
|g/dlWBC             |O           |O         |
|:                   |O           |O         |
|8000                |O           |O         |
|/mm3Plt             |O           |O         |
|:                   |O           |O         |
|300000              |O           |O         |
|/mlCreatinine       |O           |O         |
|:                   |O           |O         |
|0.5                 |O           |O         |
|mg/dlBUN            |O           |O         |
|:                   |O           |O         |
|10                  |O           |O         |
|mg/dlBeta           |O           |O         |
|hcg                 |O           |O         |
|:                   |O           |O         |
|negative            |O           |O         |
|for                 |O           |O         |
|pregnancy           |O           |O         |
|The                 |O           |O         |
|patient             |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|3-day-old           |O           |O         |
|female              |O           |O         |
|infant              |O           |O         |
|with                |O           |O         |
|jaundice            |B-DISEASE   |B-DISEASE |
|that                |O           |O         |
|started             |O           |O         |
|one                 |O           |O         |
|day                 |O           |O         |
|ago                 |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|was                 |O           |O         |
|born                |O           |O         |
|at                  |O           |O         |
|34w                 |O           |O         |
|of                  |O           |O         |
|gestation           |O           |O         |
|and                 |O           |O         |
|kept                |O           |O         |
|in                  |O           |O         |
|an                  |O           |O         |
|incubator           |O           |O         |
|due                 |O           |O         |
|to                  |O           |O         |
|her                 |O           |O         |
|gestational         |O           |B-DISEASE |
|age                 |O           |I-DISEASE |
|.                   |O           |O         |
|Vital               |O           |O         |
|signs               |O           |O         |
|were                |O           |O         |
|reported            |O           |O         |
|as                  |O           |O         |
|:                   |O           |O         |
|axillary            |O           |O         |
|temperature         |O           |O         |
|:                   |O           |O         |
|36.3°C              |O           |O         |
|,                   |O           |O         |
|heart               |O           |O         |
|rate                |O           |O         |
|:                   |O           |O         |
|154                 |O           |O         |
|beats/min           |O           |O         |
|,                   |O           |O         |
|respiratory         |O           |O         |
|rate                |O           |O         |
|:                   |O           |O         |
|37                  |O           |O         |
|breaths/min         |O           |O         |
|,                   |O           |O         |
|and                 |O           |O         |
|blood               |O           |O         |
|pressure            |O           |O         |
|:                   |O           |O         |
|65/33               |O           |O         |
|mm                  |O           |O         |
|Hg                  |O           |O         |
|.                   |O           |O         |
|Her                 |O           |O         |
|weight              |O           |O         |
|is                  |O           |O         |
|2.1                 |O           |O         |
|kg                  |O           |O         |
|,                   |O           |O         |
|length              |O           |O         |
|is                  |O           |O         |
|45                  |O           |O         |
|cm                  |O           |O         |
|,                   |O           |O         |
|and                 |O           |O         |
|head                |O           |O         |
|circumference       |O           |O         |
|32                  |O           |O         |
|cm                  |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|presents            |O           |O         |
|with                |O           |O         |
|yellow              |B-DISEASE   |O         |
|sclera              |I-DISEASE   |O         |
|and                 |O           |O         |
|icteric             |B-DISEASE   |B-DISEASE |
|body                |I-DISEASE   |I-DISEASE |
|.                   |O           |O         |
|Her                 |O           |O         |
|liver               |O           |O         |
|and                 |O           |O         |
|spleen              |O           |O         |
|are                 |O           |O         |
|normal              |O           |O         |
|to                  |O           |O         |
|palpation.Laboratory|O           |O         |
|results             |O           |O         |
|are                 |O           |O         |
|as                  |O           |O         |
|follows             |O           |O         |
|:                   |O           |O         |
|Serum               |O           |O         |
|total               |O           |O         |
|bilirubin           |O           |O         |
|:                   |O           |O         |
|21.02               |O           |O         |
|mg/dLDirect         |O           |O         |
|bilirubin           |O           |O         |
|of                  |O           |O         |
|2.04                |O           |O         |
|mg/dLAST            |O           |O         |
|:                   |O           |O         |
|37                  |O           |O         |
|U/LALT              |O           |O         |
|:                   |O           |O         |
|20                  |O           |O         |
|U/LGGT              |O           |O         |
|:                   |O           |O         |
|745                 |O           |O         |
|U/LAlkaline         |O           |O         |
|phosphatase         |O           |O         |
|:                   |O           |O         |
|531                 |O           |O         |
|U/LCreatinine       |O           |O         |
|:                   |O           |O         |
|0.3                 |O           |O         |
|mg/dLUrea           |O           |O         |
|:                   |O           |O         |
|29                  |O           |O         |
|mg/dLNa             |O           |O         |
|:                   |O           |O         |
|147                 |O           |O         |
|mEq/LK              |O           |O         |
|:                   |O           |O         |
|4.5                 |O           |O         |
|mEq/LCRP            |O           |O         |
|:                   |O           |O         |
|3                   |O           |O         |
|mg/LComplete        |O           |O         |
|blood               |O           |O         |
|cell                |O           |O         |
|count               |O           |O         |
|within              |O           |O         |
|the                 |O           |O         |
|normal              |O           |O         |
|range.She           |O           |O         |
|is                  |O           |O         |
|diagnosed           |O           |O         |
|with                |O           |O         |
|neonatal            |B-DISEASE   |B-DISEASE |
|jaundice            |I-DISEASE   |I-DISEASE |
|that                |O           |O         |
|may                 |O           |O         |
|require             |O           |O         |
|phototherapy        |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|patient             |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|53-year-old         |O           |O         |
|man                 |O           |O         |
|complaining         |O           |O         |
|of                  |O           |O         |
|frequent            |O           |O         |
|headaches           |B-DISEASE   |B-DISEASE |
|,                   |O           |O         |
|generalized         |O           |O         |
|bone                |B-DISEASE   |B-DISEASE |
|pain                |I-DISEASE   |I-DISEASE |
|and                 |O           |O         |
|difficulty          |O           |O         |
|chewing             |O           |O         |
|that                |O           |O         |
|started             |O           |O         |
|6                   |O           |O         |
|years               |O           |O         |
|ago                 |O           |O         |
|and                 |O           |O         |
|is                  |O           |O         |
|getting             |O           |O         |
|worse               |O           |O         |
|.                   |O           |O         |
|Examination         |O           |O         |
|shows               |O           |O         |
|bilateral           |O           |B-DISEASE |
|swellings           |B-DISEASE   |I-DISEASE |
|around              |O           |O         |
|the                 |O           |O         |
|molars              |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|swellings           |B-DISEASE   |B-DISEASE |
|have                |O           |O         |
|increased           |O           |O         |
|since               |O           |O         |
|his                 |O           |O         |
|last                |O           |O         |
|examination         |O           |O         |
|.                   |O           |O         |
|Several             |O           |O         |
|extraoral           |O           |O         |
|lesions             |B-DISEASE   |O         |
|of                  |I-DISEASE   |O         |
|the                 |I-DISEASE   |O         |
|head                |I-DISEASE   |O         |
|and                 |I-DISEASE   |O         |
|face                |I-DISEASE   |O         |
|are                 |O           |O         |
|detected            |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|swellings           |B-DISEASE   |O         |
|are                 |O           |O         |
|non-tender          |O           |O         |
|and                 |O           |O         |
|attached            |O           |O         |
|to                  |O           |O         |
|the                 |O           |O         |
|underlying          |O           |O         |
|bone                |O           |O         |
|.                   |O           |O         |
|Further             |O           |O         |
|evaluation          |O           |O         |
|shows               |O           |O         |
|increased           |O           |O         |
|uptake              |O           |O         |
|of                  |O           |O         |
|radioactive         |O           |O         |
|substance           |O           |O         |
|as                  |O           |O         |
|well                |O           |O         |
|as                  |O           |O         |
|an                  |O           |O         |
|increase            |O           |O         |
|in                  |O           |O         |
|urinary             |O           |O         |
|pyridinoline        |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|serum               |O           |O         |
|alkaline            |O           |O         |
|phosphatase         |O           |O         |
|is                  |O           |O         |
|300                 |O           |O         |
|IU/L                |O           |O         |
|(                   |O           |O         |
|the                 |O           |O         |
|normal              |O           |O         |
|range               |O           |O         |
|is                  |O           |O         |
|44                  |O           |O         |
|-                   |O           |O         |
|147                 |O           |O         |
|IU/L                |O           |O         |
|).                  |O           |O         |
|The                 |O           |O         |
|patient's           |O           |O         |
|sister              |O           |O         |
|had                 |O           |O         |
|the                 |O           |O         |
|same                |O           |O         |
|problems            |O           |O         |
|.                   |O           |O         |
|She                 |O           |O         |
|was                 |O           |O         |
|diagnosed           |O           |O         |
|with                |O           |O         |
|Paget's             |B-DISEASE   |B-DISEASE |
|disease             |I-DISEASE   |I-DISEASE |
|of                  |I-DISEASE   |I-DISEASE |
|bone                |I-DISEASE   |I-DISEASE |
|when                |O           |O         |
|she                 |O           |O         |
|was                 |O           |O         |
|52                  |O           |O         |
|years               |O           |O         |
|old                 |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|diagnosis           |O           |O         |
|of                  |O           |O         |
|Paget's             |O           |B-DISEASE |
|Disease             |O           |I-DISEASE |
|of                  |O           |I-DISEASE |
|Bone                |O           |I-DISEASE |
|is                  |O           |O         |
|confirmed           |O           |O         |
|and                 |O           |O         |
|Bisphosphonate      |O           |O         |
|will                |O           |O         |
|be                  |O           |O         |
|started             |O           |O         |
|as                  |O           |O         |
|first-line          |O           |O         |
|therapy             |O           |O         |
|.                   |O           |O         |
|The                 |O           |O         |
|patient             |O           |O         |
|is                  |O           |O         |
|a                   |O           |O         |
|55-year-old         |O           |O         |
|man                 |O           |O         |
|who                 |O           |O         |
|was                 |O           |O         |
|recently            |O           |O         |
|diagnosed           |O           |O         |
|with                |O           |O         |
|Parkinson's         |B-DISEASE   |B-DISEASE |
|disease             |I-DISEASE   |I-DISEASE |
|.                   |O           |O         |
|He                  |O           |O         |
|is                  |O           |O         |
|complaining         |O           |O         |
|of                  |O           |O         |
|slowness            |B-DISEASE   |B-DISEASE |
|of                  |I-DISEASE   |I-DISEASE |
|movement            |I-DISEASE   |I-DISEASE |
|and                 |O           |O         |
|tremors             |B-DISEASE   |B-DISEASE |
|.                   |O           |O         |
|His                 |O           |O         |
|disease             |O           |O         |
|is                  |O           |O         |
|ranked              |O           |O         |
|as                  |O           |O         |
|mild                |O           |O         |
|,                   |O           |O         |
|Hoehn-Yahr          |B-DISEASE   |B-DISEASE |
|Stage               |I-DISEASE   |I-DISEASE |
|I                   |I-DISEASE   |I-DISEASE |
|.                   |O           |O         |
|His                 |O           |O         |
|past                |O           |O         |
|medical             |O           |O         |
|history             |O           |O         |
|is                  |O           |O         |
|significant         |O           |O         |
|for                 |O           |O         |
|hypertension        |B-DISEASE   |B-DISEASE |
|and                 |O           |O         |
|hypercholesterolemia|B-DISEASE   |B-DISEASE |
|.                   |O           |O         |
|He                  |O           |O         |
|lives               |O           |O         |
|with                |O           |O         |
|his                 |O           |O         |
|wife                |O           |O         |
|.                   |O           |O         |
|They                |O           |O         |
|have                |O           |O         |
|three               |O           |O         |
|children            |O           |O         |
|.                   |O           |O         |
|He                  |O           |O         |
|used                |O           |O         |
|to                  |O           |O         |
|be                  |O           |O         |
|active              |O           |O         |
|with                |O           |O         |
|gardening           |O           |O         |
|before              |O           |O         |
|his                 |O           |O         |
|diagnosis           |O           |O         |
|.                   |O           |O         |
|He                  |O           |O         |
|complains           |O           |O         |
|of                  |O           |O         |
|shaking             |O           |B-DISEASE |
|and                 |O           |I-DISEASE |
|slow                |O           |I-DISEASE |
|movement            |O           |I-DISEASE |
|.                   |O           |O         |
|He                  |O           |O         |
|had                 |O           |O         |
|difficulty          |O           |O         |
|entering            |O           |O         |
|through             |O           |O         |
|a                   |O           |O         |
|door                |O           |O         |
|,                   |O           |O         |
|as                  |O           |O         |
|he                  |O           |O         |
|was                 |O           |O         |
|frozen              |O           |O         |
|and                 |O           |O         |
|needed              |O           |O         |
|guidance            |O           |O         |
|to                  |O           |O         |
|step                |O           |O         |
|in                  |O           |O         |
|.                   |O           |O         |
|His                 |O           |O         |
|handwriting         |O           |O         |
|is                  |O           |O         |
|getting             |O           |O         |
|smaller             |O           |O         |
|.                   |O           |O         |
|He                  |O           |O         |
|is                  |O           |O         |
|offered             |O           |O         |
|Levodopa            |O           |O         |
|and                 |O           |O         |
|Trihexyphenidyl     |O           |O         |
|.                   |O           |O         |
|He                  |O           |O         |
|is                  |O           |O         |
|an                  |O           |O         |
|alert               |O           |O         |
|and                 |O           |O         |
|cooperative         |O           |O         |
|man                 |O           |O         |
|who                 |O           |O         |
|does                |O           |O         |
|not                 |O           |O         |
|have                |O           |O         |
|any                 |O           |O         |
|signs               |O           |O         |
|of                  |O           |O         |
|dementia            |B-DISEASE   |B-DISEASE |
|.                   |O           |O         |
|He                  |O           |O         |
|does                |O           |O         |
|not                 |O           |O         |
|smoke               |O           |O         |
|or                  |O           |O         |
|use                 |O           |O         |
|any                 |O           |O         |
|illicit             |O           |O         |
|drugs               |O           |O         |
|.                   |O           |O         |

As we can see, hyperparameter tuning is extremely important - we achieved an f1-score of about 95%, while in our first model we had an f1-score of about 60%.

### Future

-> Stemming
