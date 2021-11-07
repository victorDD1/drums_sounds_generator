###############
###         ###
##  DATASET  ##
###         ###
###############

import numpy as np
import tensorflow as tf

def ini_dataset(data_path_npz, batch_size, labels = False):
    '''
    Initialize dataset.
    '''
    data_npz = np.load(data_path_npz)
    categories = data_npz['categories']

    if labels:
        labels = data_path_npz['labels']
        dataset = tf.data.Dataset.from_tensor_slices([data_npz['data'], data_npz['labels']]).batch(batch_size)
    else :
        dataset = tf.data.Dataset.from_tensor_slices(data_npz['data']).batch(batch_size)

    return dataset, categories