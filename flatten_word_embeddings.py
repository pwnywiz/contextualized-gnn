import pickle
import os
import numpy as np


# A script that accepts a dictionary with a list of embeddings
# for a word as in word = [e_1, e_2,...,e_N] and converts
# it to a flattened dictionary of the format word_N = embed
read_file_name = 'embed_dict'
write_file_name = 'embed_dict_flattened'

print('Reading dictionary from file')
f = open("{}.pkl".format(read_file_name),"rb")
embed_dict = pickle.load(f)
f.close()

embed_dict_flattened = dict()

print('Flattening dictionary')
for k in embed_dict:
    for count, e in enumerate(embed_dict[k]):
        embed_dict_flattened['{}_{}'.format(k, count)] = e

if os.path.exists('{}.pkl'.format(write_file_name)):
    print('Removing old file')
    os.remove('{}.pkl'.format(write_file_name))
print('Writing dictionary to file')
new_file = open("{}.pkl".format(write_file_name),"wb")
pickle.dump(embed_dict_flattened,new_file)
new_file.close()