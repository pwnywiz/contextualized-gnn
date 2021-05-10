from transformers import BertModel, AutoTokenizer
import torch
import pickle
import os
import numpy as np
from utils import *

# - Accept a file with a document in each row
# - Parse each document by sentence
# - Tokenize each sentence word into subwords through BERT tokenizer
# - Open the dictionary with the precomputed embedding clusters (up to 4 for each subword)
# - Use euclidian distance to match every subword's embedding to the closest of the precomputed embeddings
# - Add a suffix to the subword '_N' where N is the index of the matched embedding from the list
# - Write each document back to a file

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print(torch.cuda.device_count())
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.get_device_name(1))

PRE_TRAINED_MODEL_NAME = 'nlpaueb/bert-base-greek-uncased-v1'

tokenizer = AutoTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)
model = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)

model = model.to(device)

model.eval()

# Load embeddings dictionary from file
print('Reading dictionary from file')
embed_file_name = 'embed_dict'
f = open("{}.pkl".format(embed_file_name), "rb")
output = pickle.load(f)

print('Reading text from file')
text_file_name = 'search_results'
sr = open("{}.txt".format(text_file_name), "r", encoding="utf8")

def get_euclidian_distance_mapping(subword, subword_embedding):
  if subword not in output:
    print('{} not in dictionary'.format(subword))
    return '{}_0'.format(subword)
  node_num = closest_node(subword_embedding.numpy(), output[subword])
  return '{}_{}'.format(subword, str(node_num))

def map_subwords_into_centroids(document, tokenizer, model, device, truncate_len = -1):
  mapped_words = []
  if truncate_len > 0:
    words = document.split(' ')[:truncate_len]
    document = ' '.join(words)
  splitted_sentences = sent_tokenize(document)
  for sentence in splitted_sentences:
    sentence_ids = encode_sentence(sentence, tokenizer)
    hidden_states = sentence_inference(sentence_ids, model, device)
    last_hidden_state_cleaned = get_last_hidden_state_cleared(hidden_states)
    cleaned_sentence = get_tokens_cleared(sentence_ids, tokenizer)

    for i in range(len(last_hidden_state_cleaned)):
      mapped_word = get_euclidian_distance_mapping(cleaned_sentence[i], last_hidden_state_cleaned[i])
      mapped_words.append(mapped_word)
  
  return ' '.join(mapped_words)

new_file_name = 'search_results_bert'

if os.path.exists('{}.txt'.format(new_file_name)):
    print('Removing old file')
    os.remove('{}.txt'.format(new_file_name))
new_file = open('{}.txt'.format(new_file_name), "w", encoding='utf8')

print('Writing mapped subwords to file')
for line in sr.readlines():
  new_line = line.strip()
  mapped_sentence = map_subwords_into_centroids(new_line, tokenizer, model, device, truncate_len = 450)
  new_file.write(mapped_sentence)
  new_file.write('\n')

f.close()
sr.close()
new_file.close()