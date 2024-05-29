import numpy as np 
import glob 
import pickle
import matplotlib.pyplot as plt  
import functions
import scipy.io as sio
from lib import proise_v2
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict, GridSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score
from sklearn.decomposition import PCA, FastICA, KernelPCA, SparsePCA, TruncatedSVD
from sklearn.linear_model import RidgeClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler

# fileList = glob.glob("./stmtf/*.pkl")
tabStrf  = []
tabSession = []
tabDaySession = []
tabSubjectNb = []
tabFileNb = []
outFolder = './out_01_acoustic_analysis/'

dataFolder = './out_00_data/'

#load data
data = pickle.load(open(str(dataFolder)+'data.pkl', 'rb'))
tabStrf = data['tabStrf']
tabStrf = tabStrf 
tabSession = data['tabSession']
tabSubjectNb = data['tabSubjectNb']
tabTask = data['tabTask']

tabStrf = np.asarray(tabStrf)
tabSession = np.asarray(tabSession)
tabSubjectNb = np.asarray(tabSubjectNb)
tabDaySession = np.asarray(tabDaySession)

nbChannels = 128
nbRates = 14
nbScales = 8

# per subject
tabIndex = np.unique(tabSubjectNb)
allCanonicalMaps = []
training_accuracies_mean = []
training_accuracies_std  = []
testing_accuracies_mean  = []
testing_accuracies_std   = []
for index in tabIndex:
  print('Subject nb '+str(index))
  X = []
  Y = []
  X = tabStrf[tabSubjectNb==(index)][:]
  Y = tabSession[tabSubjectNb==(index)]
  uniqueSessions = np.unique(Y)
  # print(uniqueSessions)

  # print averaged strf before and after and diff between
  mean_before_tcc = (np.nanmean(np.asarray(X[Y=='1'][:]),axis=0))
  mean_after_tcc = (np.nanmean(np.asarray(X[Y=='2'][:]),axis=0))

  diff_mean = np.divide(np.abs(mean_after_tcc-mean_before_tcc),(mean_before_tcc+mean_after_tcc)/2)
  strf_session_mean_reshaped = np.reshape(diff_mean,(nbChannels,nbScales,nbRates))    
  strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(strf_session_mean_reshaped),nbChannels=128,nbRates=14,nbScales=8)
  functions.plotStrfavg(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname=outFolder+'subject#'+str(index)+'_diff_before_after_tcc',show='false')
  
  for iUniqueSession in uniqueSessions:
    X_session_mean = np.nanmean(X[Y==iUniqueSession],axis=0)
    strf_session_mean_reshaped = np.reshape(X_session_mean,(nbChannels,nbScales,nbRates))    
    strf_scale_rate, strf_freq_rate, strf_freq_scale = functions.avgvec2strfavg(functions.strf2avgvec(strf_session_mean_reshaped),nbChannels=128,nbRates=14,nbScales=8)
    functions.plotStrfavg(strf_scale_rate, strf_freq_rate, strf_freq_scale,aspect_='auto', interpolation_='none',figname=outFolder+'subject#'+str(index)+'_session#'+iUniqueSession,show='false')

  ##### First strategy
  # classification: train on session 1 and 3 and test on session 2 and 4
  print("First strategy: train on 1 and 3 and test on 2 and 4")
  X_train = np.concatenate((X[Y=='1'],X[Y=='3']))
  y_train = np.concatenate((Y[Y=='1'],Y[Y=='3']))
  X_test  = np.concatenate((X[Y=='2'],X[Y=='4']))
  y_test  = np.concatenate((Y[Y=='2'],Y[Y=='4']))
  
  y_train[y_train=='1'] = 0
  y_train[y_train=='3'] = 1    
  y_test[y_test=='2']   = 0
  y_test[y_test=='4']   = 1

  scaler = RobustScaler(quantile_range=(25.0, 75.0))
  scaler.fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)

  # print(X_train.shape)
  # print(y_test.shape)
  n_cv = 5
  cv = StratifiedKFold(n_cv, shuffle=True)
  
  n_components = 40
  print('____number of PCA components: '+str(n_components))
  pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
  X_train_pca = pca.transform(np.asarray(X_train))
  print('____PCA explained variance : '+str(np.sum(pca.explained_variance_ratio_)))
  
  # grid search classifier definition and fit
  tuned_parameters = { 'gamma':np.logspace(-3,3,num=30),'C':np.logspace(-3,3,num=30)}
  clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=-1, cv=cv,  pre_dispatch=6,
                 scoring='balanced_accuracy', verbose=False)
  # clf.fit(X_train_pca, y_train)
  clf = RidgeClassifierCV(alphas=[1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]).fit(X_train_pca, y_train)
  # print('GridSearch done')

  y_pred = clf.predict(pca.transform(X_train))
  print("____training accuracy: %0.2f" % (balanced_accuracy_score(y_train,y_pred)))
  y_pred = clf.predict(pca.transform(X_test))  
  print("____testing accuracy: %0.2f" % (balanced_accuracy_score(y_test,y_pred)))
  print()

  ##### Second strategy
  # classification: train / test split with 5-fold cv
  print("Second strategy: 5-fold cross val on all samples (same as PLoS Comp Biol)")
  n_fold = 10
  accTest = []
  accTrain = []
  canonicalMapsFolds = []
  for fold in range(n_fold):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=fold)
    y_train[y_train=='1'] = 0
    y_train[y_train=='2'] = 0    
    y_train[y_train=='3'] = 1
    y_train[y_train=='4'] = 1   

    y_test[y_test=='1']   = 0
    y_test[y_test=='2']   = 0    
    y_test[y_test=='3']   = 1
    y_test[y_test=='4']   = 1    

    print(np.unique(y_train))
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)   
    X_train_pca = pca.transform(np.asarray(X_train))

    # grid search classifier definition and fit
    tuned_parameters = { 'gamma':np.logspace(-3,3,num=3),'C':np.logspace(-3,3,num=3)}
    clf = GridSearchCV(SVC(kernel='rbf'), tuned_parameters, n_jobs=-1, cv=cv,  pre_dispatch=6,
                   scoring='balanced_accuracy', verbose=False)
    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(pca.transform(X_train))
    accTrain.append(balanced_accuracy_score(y_train,y_pred))
    y_pred = clf.predict(pca.transform(X_test))    
    accTest.append(balanced_accuracy_score(y_test,y_pred))

    ################################################################################################################################
    # interpretation stim+pseudo-noise
    N = 5
    dimOfinput = (128,8,14) # dimension of the input representation
    probingMethod = 'revcor' # choice of the probing method : bubbles or revcor
    samplesMethod = 'pseudoRandom' # choice of the method to generate probing samples trainSet or pseudoRandom (need x_test_set) or gaussianNoise
    nbRevcorTrials = X_train.shape[0]*N # number of probing samples for the reverse correlation (must be below the number of training sample if trainSet)
    nDim_pca = 30 # number of dimension to compute the PCA for the pseudo-random noise generation
    probingSamples, _ = proise_v2.generateProbingSamples(x_train_set = X_train, x_test_set = X_train, dimOfinput=dimOfinput, probingMethod = probingMethod, samplesMethod = samplesMethod, nDim_pca = nDim_pca, nbRevcorTrials = nbRevcorTrials)
    print(probingSamples.shape)
    X_probingSamples_pca = pca.transform(probingSamples/2 + np.tile(X_train,(N,1))/2)  
    y_pred = clf.predict(X_probingSamples_pca)
    print(np.unique(y_pred))
    responses_ = np.squeeze((y_pred == np.tile(y_train,(1,N))) & (np.tile(y_train,(1,N)) == '1'))

    canonicalMap = []
    pval = []

    data2revcor = np.asarray(probingSamples)

    canonicalMap = np.nanmean(data2revcor[responses_][:],axis=0) - np.nanmean(data2revcor[np.logical_not(responses_)][:],axis=0)
    canonicalMapsFolds.append(canonicalMap)
  allCanonicalMaps.append(np.mean(np.asarray(canonicalMapsFolds),axis=0))
  print('____training accuracy: '+str(np.nanmean(accTrain))+'+/-'+str(np.nanstd(accTrain)))    
  print('____testing accuracy: '+str(np.nanmean(accTest))+'+/-'+str(np.nanstd(accTest)))
  print()

  training_accuracies_mean.append(np.nanmean(accTrain))
  testing_accuracies_mean.append(np.nanmean(accTest))
  training_accuracies_std.append(np.nanstd(accTrain))
  testing_accuracies_std.append(np.nanstd(accTrain))

sio.savemat(outFolder+'withinCanonicalMaps.mat', {'allCanonicalMaps': np.asarray(allCanonicalMaps), 'training_accuracies_mean':np.asarray(training_accuracies_mean), 'training_accuracies_std':np.asarray(training_accuracies_std), 'testing_accuracies_mean':np.asarray(testing_accuracies_mean), 'testing_accuracies_std':np.asarray(testing_accuracies_std)})
pickle.dump({'allCanonicalMaps': np.asarray(allCanonicalMaps), 'training_accuracies_mean':np.asarray(training_accuracies_mean), 'training_accuracies_std':np.asarray(training_accuracies_std), 'testing_accuracies_mean':np.asarray(testing_accuracies_mean), 'testing_accuracies_std':np.asarray(testing_accuracies_std)}, open(outFolder+'withinCanonicalMaps.pkl', 'wb'))




