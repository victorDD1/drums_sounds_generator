#####################
###               ###
##  VISUALISATION  ##
###               ###
#####################

import numpy as np
import os
import matplotlib.pyplot as plt
import librosa
import IPython.display as ipd
from IPython.core.display import display

from data.preprocessing import normalize

def selectRandomSoundPerCat(categories, classifiedSoundPath):
  soundsPaths = []
  for cat in categories:
    soundsPaths.append(classifiedSoundPath+'/'+cat+'/'+np.choice(os.listdir(classifiedSoundPath+'/'+cat)))
  return soundsPaths

def plotSoundsFromPath(soundsPaths):
  
  n_display = len(soundsPaths)
  if n_display%2 != 0:
    n_display-=1
  for i in range(n_display):
    signal, sr = librosa.load(soundsPaths[i], sr = None)
    if np.size(signal[0])>1:
      signal = signal[:,0]
    filename = soundsPaths[i].split('/')[-1]
    signal = normalize(signal)

    fig= plt.figure(figsize=(25,10))

    plt.subplot(n_display//2,2,i+1)
    plt.title(filename)
    plt.plot(signal, color='gray')
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
    display(ipd.Audio(signal, rate=sr))

