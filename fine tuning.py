
import torch
torch.cuda.empty_cache()
import gc

gc.collect()
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample

import logging
from datetime import datetime
import sys
import os
import gzip
import csv
import pandas as pd
import xlrd




sts_dataset_path =path_to_dataset



model_name='multilingualBert'
# Read the dataset
train_batch_size = 16
num_epochs = 4
model_save_path = '/training_stsbenchmark_'+model_name.replace("/", "-")+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")



word_embedding_model = models.Transformer(model_name,max_seq_length=256)


pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

logging.info("Read STSbenchmark train dataset")

train_samples = []
dev_samples = []
test_samples = []
samplePercent=1
xls = pd.ExcelFile(sts_dataset_path)
df1 = pd.read_excel(xls, 'train')
df1 = df1.sample(frac=1).reset_index(drop=True)
df2 = pd.read_excel(xls, 'dev')
df2 = df2.sample(frac=1).reset_index(drop=True)
df3=pd.read_excel(xls, 'test')
df3 = df3.sample(frac=1).reset_index(drop=True)
for index, row in df1.iterrows():
  score = float(row['score']) / 5.0
  inp_example=InputExample(texts=[row['sent1Fa'], row['sent1']], label=score)
  train_samples.append(inp_example)
for index, row in df2.iterrows():
  score = float(row['score']) / 5.0
  inp_example=InputExample(texts=[row['sent1Fa'], row['sent1']], label=score)
  dev_samples.append(inp_example)
for index, row in df3.iterrows():
  score = float(row['score']) / 5.0
  inp_example=InputExample(texts=[row['sent1Fa'], row['sent1']], label=score)
  test_samples.append(inp_example)

print('end of reading data')
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')



warmup_steps = math.ceil(len(train_dataloader) * num_epochs  * 0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))



model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)



model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)