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

  n_components = 30
  n_cv = 5
  cv = StratifiedKFold(n_cv, shuffle=True)
  ##### 1 vs. 2
  # classification: train / test split with 5-fold cv
  print("1 vs. 2")
  n_fold = 10
  accTest = []
  accTrain = []
  canonicalMapsFolds = []
  for fold in range(n_fold):
    X_12 = np.concatenate((X[Y=='1'],X[Y=='2']))
    Y_12 = np.concatenate((Y[Y=='1'],Y[Y=='2']))

    X_train, X_test, y_train, y_test = train_test_split(X_12, Y_12, test_size=0.25, random_state=fold)
    y_train[y_train=='1'] = 1
    y_train[y_train=='2'] = 2    

    y_test[y_test=='1']   = 1
    y_test[y_test=='2']   = 2    

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
    y_pred = clf.predict(pca.transform(X_train))

  print("____training accuracy: %0.2f" % np.nanmean(accTrain))
  y_pred = clf.predict(pca.transform(X_test))  
  print("____testing accuracy: %0.2f" % np.nanmean(accTest))
  print()    

  training_accuracies_mean.append(np.nanmean(accTrain))
  testing_accuracies_mean.append(np.nanmean(accTest))
  training_accuracies_std.append(np.nanstd(accTrain))
  testing_accuracies_std.append(np.nanstd(accTrain))

  ##### 3 vs. 4
  # classification: train / test split with 5-fold cv
  print("3 vs. 4")
  n_fold = 10
  accTest = []
  accTrain = []
  canonicalMapsFolds = []
  for fold in range(n_fold):
    X_34 = np.concatenate((X[Y=='3'],X[Y=='4']))
    Y_34 = np.concatenate((Y[Y=='3'],Y[Y=='4']))

    X_train, X_test, y_train, y_test = train_test_split(X_34, Y_34, test_size=0.25, random_state=fold)
    y_train[y_train=='3'] = 3
    y_train[y_train=='4'] = 4    

    y_test[y_test=='3']   = 3
    y_test[y_test=='4']   = 4 

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
  print("____training accuracy: %0.2f" % np.nanmean(accTrain))
  y_pred = clf.predict(pca.transform(X_test))  
  print("____testing accuracy: %0.2f" % np.nanmean(accTest))
  print()    




