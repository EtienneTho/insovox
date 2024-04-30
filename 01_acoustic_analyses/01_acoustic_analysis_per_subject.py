import numpy as np 
import glob 
import pickle
import matplotlib.pyplot as plt  
import functions
import scipy.io as sio
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
  n_cv = 10
  cv = StratifiedKFold(n_cv, shuffle=True)
  
  n_components = 10
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
  # classification: train on session 1 and 3 and test on session 2 and 4
  print("Second strategy: 5-fold cross val on all samples (same as PLoS Comp Biol)")
  n_fold = 10
  accTest = []
  accTrain = []
  for fold in range(n_fold):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=fold)

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

  print('____training accuracy: '+str(np.nanmean(accTrain))+'+/-'+str(np.nanstd(accTrain)))    
  print('____testing accuracy: '+str(np.nanmean(accTest))+'+/-'+str(np.nanstd(accTrain)))

  print()



