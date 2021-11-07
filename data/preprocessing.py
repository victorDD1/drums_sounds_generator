#####################
###               ###
##  PREPROCESSING  ##
###               ###
#####################

### Preprocessing utility function for waveform preprocessing

import numpy as np

def adjust_sample_length(sample, sample_length, sr):
    '''
    Adjust length of the considered sample.
        -> crop if too long
        -> zero pad (right) if too short
    '''
    size = int(sample_length*sr)
    if np.size(sample)>=size:
        return sample[:size]
    else:
        N = int(size - np.size(sample))
        return np.concatenate((sample, np.zeros((N))), axis = 0)

def normalize(sample):
    m = max(sample)
    if m != 0:
        return sample/m

