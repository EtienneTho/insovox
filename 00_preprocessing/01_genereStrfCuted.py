'''
Copyright (c) Etienne Thoret
All rights reserved

'''

import matplotlib.pylab as plt
import auditory
import utils
import pickle
import numpy as np
import os
import glob
import scipy.io as sio
from joblib import Parallel, delayed

def strfCuted(filename):
    durationSegment = 15
    mypath = './stmtf/'
    audio, fs = utils.audio_data(filename)
    lengthSegmentSample = np.floor(durationSegment*fs)
    nbSegment = int(len(audio) / lengthSegmentSample)

    for iSegment in range(nbSegment):   
        print()
        print(filename+' || segment nb '+str(iSegment)+'/'+str(nbSegment))
        filename_splited = filename.split("_")
        print(filename_splited)
        idSubject = filename_splited[1]
        idSession = filename_splited[2]
        idTask    = filename_splited[3].split(".")[0]

        indMin = int(lengthSegmentSample*iSegment)
        indMax = min(len(audio),int(lengthSegmentSample*(iSegment+1)))
        strf, auditory_spectrogram_, mod_scale, scale_rate = auditory.strf(audio[indMin:indMax],audio_fs=fs, duration=-1)
        strf_avgTime = np.mean(np.abs(strf),axis=0)
        print(mypath)
        pickle.dump({'strf': strf_avgTime,
                     'auditory_spectrogram':auditory_spectrogram_,
                     'idSubject': idSubject,
                     'idTask': idTask,
                     'idSession' : idSession,
                     'iSegment': iSegment,
                     'durationSegment': durationSegment,
                     'filename': filename,
                     'fs': fs,
                     'nbSegment': nbSegment,
                     'lengthAudioSample': len(audio)
        }, open(os.path.join(mypath+'strf_idSession_'+str(idSession)+'_idSubject_'+str(idSubject)+'_idTask_'+str(idTask)+'_segmentNb_'+str(iSegment)+'.pkl'), 'wb'))

# print(glob.glob("../../04_monoCutCutCleanAiff/*.aiff"))

input_path = './originalRecordings/'
#strfCuted('./originalRecordings/2024_02_2.aiff')
Parallel(n_jobs=3, verbose=1, backend="loky")(delayed(strfCuted)(filename) for filename in glob.glob(input_path+"*.aiff"))



