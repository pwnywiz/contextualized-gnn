import torch
import pickle
import os
from sklearn.cluster import KMeans
import numpy as np

# import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def save_dict_to_file(embed_dict={}, file_name='embed_dict', read_file=False):
  if os.path.exists('{}.pkl'.format(file_name)):
    print('Removing old file')
    os.remove('{}.pkl'.format(file_name))
  print('Writing dictionary to file')
  f = open("{}.pkl".format(file_name),"wb")
  pickle.dump(embed_dict,f)
  f.close()

  if read_file:
    print('Reading dictionary from file')
    f = open("{}.pkl".format(file_name),"rb")
    output = pickle.load(f)
    for k in output:
      print("{}: {}".format(k, len(output[k])))
    f.close()

def flatten_list(_2d_list):
  flat_list = []
  # Iterate through the outer list
  for element in _2d_list:
    if type(element) is list:
      # If the element is of type list, iterate through the sublist
      for item in element:
        flat_list.append(item)
    else:
      flat_list.append(element)
  return flat_list

def encode_sentence(sentence, tokenizer):
  sentence_ids = tokenizer.encode(sentence)
  if len(sentence_ids) > 512:
      sentence_ids = sentence_ids[0:510]
      sentence_ids.append(102)
    #   print(sentence_ids)
  # print('sentence_ids', sentence_ids)
  # print('type of sentence_ids', type(sentence_ids))
  # print('sentence_tokens', tokenizer.convert_ids_to_tokens(sentence_ids))
  return torch.LongTensor(sentence_ids)

def sentence_inference(sentence_ids, model, device):
  device_sentence_ids = sentence_ids.to(device)
  device_sentence_ids = device_sentence_ids.unsqueeze(0)
  with torch.no_grad():
    out = model(input_ids=device_sentence_ids)
  del device_sentence_ids
  l = [x.cpu() for x in out[2]]
  del out
  return l

def update_embed_dict(embed_dict, key, value):  
  if key in embed_dict.keys(): 
    embed_dict[key].append(value)
  else:
    new_item = {key: [value]}
    embed_dict.update(new_item)

def get_last_hidden_state_cleared(hidden_states):
  last_hidden_state = hidden_states[-1].squeeze()
  return last_hidden_state[1:-1]

def get_tokens_cleared(sentence_ids, tokenizer):
  original_sentence = tokenizer.convert_ids_to_tokens(sentence_ids)
  return original_sentence[1:-1]

# - Truncate document
# - Split into sentences
# - Encode every sentence
# - Add embeddings into dictionary
def embed_document_by_sentence(document, embed_dict, tokenizer, model, device, truncate_len = -1):
  if truncate_len > 0:
    words = document.split(' ')[:truncate_len]
    # print(len(words))
    document = ' '.join(words)
  splitted_sentences = sent_tokenize(document)
  for sentence in splitted_sentences:
    sentence_ids = encode_sentence(sentence, tokenizer)
    hidden_states = sentence_inference(sentence_ids, model, device)
    last_hidden_state_cleaned = get_last_hidden_state_cleared(hidden_states)
    # print(last_hidden_state_cleaned)
    cleaned_sentence = get_tokens_cleared(sentence_ids, tokenizer)

    for i in range(len(last_hidden_state_cleaned)):
      update_embed_dict(embed_dict, cleaned_sentence[i], last_hidden_state_cleaned[i])

def kmeans_word_embeddings(embed_pt_array, embed_clusters=4):
  embed_array = [x.numpy() for x in embed_pt_array]
  if len(embed_array) <= embed_clusters:
    return embed_array
  kmeans = KMeans(n_clusters=embed_clusters, random_state=0).fit(embed_array)
  centers = np.array(kmeans.cluster_centers_)
  return centers

# Used for calculating the euclidian distance for a
# new point and the closest centroid in a cluster
def closest_node(node, nodes):
  if len(nodes) == 1:
    return 0
  nodes = np.asarray(nodes)
  dist_2 = np.sum((nodes - node)**2, axis=1)
  return np.argmin(dist_2)