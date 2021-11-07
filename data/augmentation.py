####################
###              ###
##  AUGMENTATION  ##
###              ###
####################

import numpy as np
import random as rd

def data_augmentation(waveform):

    # Temporal offset (zero padding)
    size = waveform.shape[-1]
    offset_factor = 15
    offset = rd.uniform(1, size//offset_factor)
    waveform = np.concatenate((np.zeros((1,1,offset)), waveform), axis = -1)

    # Phase +180
    if rd.randint(0,1):
        waveform = - waveform

    return waveform

def daya_augmentation_batch(batch):
  for i in range(len(batch)):
    batch[i]=data_augmentation(batch[i])
  return batch