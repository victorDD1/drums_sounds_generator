######################
###                ###
##  BUILD DATABASE  ##
###                ###
######################

import argparse
import os
from os.path import join as j
import numpy as np
import shutil
import glob
import librosa
import tqdm

from data.preprocessing import adjust_sample_length, normalize

nameDrumsSounds = ['kick', 'hihat', 'snare', 'percution', 'tom', 'FX', 'ride', 'clap', 'others']

dictNameDrumsSounds = {
    'kick' : ['kick', 'kik', 'kk', 'bd'],
    'hihat' : ['hihat', 'hh', 'hat', 'hi_hat', 'hi', 'open', 'oh', 'ch'],
    'snare' : ['snare', 'snr', 'sd'],
    'percution' : ['percution', 'perc', 'wood', 'shaker', 'conga', 'stick', 'maracas', 'bell'],
    'tom' : ['tom','toms'],
    'FX' : ['fx', 'effects', 'effect', 'cowbell'],
    'ride' : ['ride', 'crash', 'crsh', 'cymbal'],
    'clap' : ['clap', 'claps', 'cla'],
    'others' : []
    }

def exploreFolder(dataFolderPath, listFilesPaths):
  '''recursive function that list paths in a directory'''
  soundFilesPaths = glob.glob(dataFolderPath+'/'+'*.wav')
  if len(soundFilesPaths)!=0 :
    #print(soundFilesPaths)
    listFilesPaths += soundFilesPaths
  else:
    for path in os.listdir(dataFolderPath):
      exploreFolder(dataFolderPath+'/'+path, listFilesPaths)

  return listFilesPaths

def classification(filename, targetNames):
  '''classify samples in categories depending on their names'''
  categ = 'others'
  filenameLower = filename.lower()
  for nameSounds in targetNames.keys():
    if any([(otherNames in filenameLower) for otherNames in targetNames[nameSounds]]) :
      categ = nameSounds
      break
  return categ


def soundsClassification(data_path, data_pathSave):
  '''classify samples in a classified data directory according to their categrories'''
  print(os.listdir(data_path))
  filesPaths = exploreFolder(data_path, [])
  #print('Paths for Classification :', filesPaths)

  for filePath in filesPaths :
    fileName = filePath.split('/')[-1]
    fileName = filePath.split('\\')[-1]

    categ = classification(fileName, dictNameDrumsSounds)
    saveDirPath = data_pathSave+'/'+categ
    if not os.path.exists(saveDirPath):
      os.makedirs(saveDirPath)
    shutil.copy(filePath, saveDirPath+'/'+fileName)
      
def data_as_numpyarray(data_path, categorie, sample_length, sr):
    '''
    Save data as numpy array. Preprocess data (normalise, duration). Size of the numpy array : [N,T,1]
    '''
    T = int(sample_length*sr)
    errors = 0
    data = np.empty(0)
    for path in tqdm.tqdm(os.listdir(j(data_path, categorie)),categorie):
        soundPath = os.path.join(data_path, categorie, path)

        try :
            sample = librosa.load(soundPath, sr, mono=True, duration=sample_length)[0] # [C, T]
        except :
            errors += 1
            continue

        if sample_length != sample.shape[0]:
            sample = adjust_sample_length(sample, sample_length, sr)

        sample = normalize(sample)

        try :
            sample = np.reshape(sample,(1, T, 1)) # Pre batch sample [1,T,1]
        except :
            errors += 1
            continue

        if data.shape[0]== 0:
            data = sample
        else :
            data = np.concatenate((data, sample), axis = 0) # [N,T,1] with N the number of samples

    if not os.path.exists(data_path+'/'+categorie+'.npy'):
        np.save(j(data_path, f'{categorie}_{sr//1000}kHz.npy'), data)

    print(f'Number of errors : {errors}')
    print(f'Number of {categorie} samples : {data.shape[0]}')

def save_as_npz(data_path, categories_list, sr):
  '''
  Save data as one compressed array at data_path.
  for each categories in categories_list data['<categories>'] contains sample of the categorie and the label.
  '''
  data = np.empty(0)
  labels = np.empty(0)
  fn = 'data_'

  for i, cat in tqdm.tqdm(enumerate(categories_list), 'categories'):
    # Load arrays
    array_path = j(data_path, f'{cat}_{sr//1000}kHz.npy')

    if data.shape[0] == 0:
      data = np.load(array_path)
      labels = np.array([i]).repeat(data.shape[0])

    else :
      arr = np.load(array_path)
      data = np.concatenate((data, arr), axis = 0)
      labels = np.concatenate((labels, np.array([i]).repeat(arr.shape[0])), axis = 0)

    fn = fn + f'{cat}_'
  
  #Save Data
  fn = fn + f'{sr//1000}kHz.npz'
  np.savez(j(data_path, fn), data=data, labels=labels, categories=categories_list)
  print(f'Data saved at {j(data_path, fn)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build database')

    parser.add_argument('-dp', '--data_path', help='Path of database directory', type=str)
    parser.add_argument('-svp', '--saving_data_path', help='Path of database directory', type=str)
    parser.add_argument('-sl', '--sample_length', help='Length of the audio samples (in s)', type=float)
    parser.add_argument('-sr', '--sampling_rate', help='Sampling rate of the audio samples (in Hz)', type=int)
    parser.add_argument('-save_numpy_only', help='Save classified data as numpy arrays', default=False, action='store_true')

    args = parser.parse_args()
    if not(args.save_numpy_only):
        soundsClassification(args.data_path, args.saving_data_path)
        data_as_numpyarray(args.saving_data_path, args.sample_length, args.sampling_rate)
    else :
        for categorie in nameDrumsSounds:
            data_as_numpyarray(args.saving_data_path, categorie, args.sample_length, args.sampling_rate)