import numpy as np 
import glob 
import pickle
import functions
import matplotlib.pyplot as plt 

fileList = glob.glob("../00_preprocessing/stmtf/*.pkl") #stmtf
tabStrf  = []
tabRates = [] 
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabSegment = []
tabStanford = []
tabFilename = []
tabSegment = []
tabSR = []
tabFR = []
tabFS = []
tabTask = []
outFolder = './out_00_data/'

# load data
for iFile, filename in enumerate(sorted(fileList)):
  print(str(iFile+1)+'/'+str(len(fileList)))
  dataFor = pickle.load(open(filename, 'rb'))
  tabFilename.append(filename)
  print(filename)

  toAdd = dataFor['strf']
  toAdd = toAdd / np.amax(toAdd)

  print(toAdd.shape)
  tabStrf.append(toAdd.flatten())#/np.sum(dataFor['strf'].flatten())) # !!!!! NORMALISATION
  strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(toAdd, abs_=False),nbChannels=128,nbRates=14,nbScales=8)
  tabSR.append(strf_scale_rate.flatten())
  tabFR.append(strf_freq_rate.flatten())
  tabFS.append(strf_freq_scale.flatten())

  tabSession.append(dataFor['idSession'])
  tabSubjectNb.append(int(dataFor['idSubject']))
  tabTask.append(dataFor['idTask'])
  
  print()
  print(dataFor['durationSegment'])
  print(int(dataFor['idSubject']))

  tabSegment.append(int(dataFor['iSegment']))

#save data
pickle.dump({'tabTask': tabTask, 'tabSR':tabSR, 'tabFR': tabFR, 'tabFS': tabFS, 'tabStrf': tabStrf, 'tabSegment': tabSegment, 'tabSession': tabSession, 'tabSubjectNb': tabSubjectNb}, open(str(outFolder)+'data.pkl', 'wb'))


