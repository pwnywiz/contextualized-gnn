import torch
from transformers import BertModel, AutoTokenizer
import numpy as np
import pandas as pd
from utils import embed_document_by_sentence, save_dict_to_file, kmeans_word_embeddings

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(torch.cuda.device_count())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(0))

df = pd.read_csv('search_results.csv')
# print(df.head())

print(df.shape)
df = df[df['text'].map(lambda x: len(x.split()) < 10000)]
df = df.drop_duplicates(subset=['text'])
print(df.shape)

PRE_TRAINED_MODEL_NAME = 'nlpaueb/bert-base-greek-uncased-v1'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)

model = model.to(device)

model.eval()

embed_dict = dict()
centroid_dict = dict()

for row in df.text:
  embed_document_by_sentence(row, embed_dict, tokenizer, model, device, truncate_len = 450)

print(len(embed_dict))

for k in embed_dict:
    centers = kmeans_word_embeddings(embed_dict[k])
    centroid_dict[k] = centers
save_dict_to_file(embed_dict=centroid_dict, read_file=False)
