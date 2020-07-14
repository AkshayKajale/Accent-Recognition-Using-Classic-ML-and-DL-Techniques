# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

import multiprocessing
import librosa

from src import getsplit


from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt

from itertools import cycle

RATE = 24000
COL_SIZE = 100

def get_wav(language_num):
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def get_features(wav):
    #sc = librosa.feature.spectral_centroid(y=wav, sr=RATE, hop_length=int(RATE/40), n_fft=int(RATE/40))
    #zrc = librosa.feature.zero_crossing_rate(y=wav, frame_length=int(RATE/40), hop_length=int(RATE/40))
    return librosa.feature.mfcc(y=wav, sr=RATE, hop_length=int(RATE/40), n_fft=int(RATE/40), n_mfcc=13)
    #features = np.append(mfcc, np.append(sc, zrc, axis=0), axis=0)
    #return features

def segments(mfccs,labels):
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label) 
    return(segments, seg_labels)

def knn_classifier(X, Y):
    print("Shape of X", X.shape)
    
    candidate_acc = []
    neighbors = [50, 75, 100, 125]
    p = [1, 2]
    
    for n in neighbors:
        for dist in p:
            classifier = KNeighborsClassifier(n_neighbors=n, metric='minkowski', p=dist)
            cv = KFold(n_splits=10, shuffle=True)
    
            for train_index, test_index in cv.split(X):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index] 
                classifier.fit(X_train, Y_train)
       
            scores = cross_val_score(classifier, X, Y, cv=10)
            best_acc = np.max(np.array(scores))
            candidate_acc.append(best_acc)
            print('For neighbors = %d and distance type = %d, best 10 fold cross validation accuracy = %f' % (n, dist, best_acc))
                   
    return classifier
    
def rf_classifier(X, Y):
    print("Shape of X", X.shape)
    
    candidate_acc = []
    n_tree = [500, 750, 1000, 1250]
    
    for n in n_tree:
        classifier = RandomForestClassifier(n_estimators = n, criterion = 'entropy', random_state = 0)
        cv = KFold(n_splits=10, shuffle=True)
    
        for train_index, test_index in cv.split(X):
            X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index] 
            classifier.fit(X_train, Y_train)
       
        scores = cross_val_score(classifier, X, Y, cv=10)
        best_acc = np.amax(scores)        
        candidate_acc.append(best_acc)
        print('For number of trees = %d, best 10 fold cross validation accuracy = %f' % (n, best_acc))       
    return classifier

def ovr_classifier(X, Y):
    print("Shape of X", X.shape)
    
    candidate_acc = []
    C = [1,2,3,4,5]
    kernel = ['linear','rbf']
    
    for C in C:
        for kernel in kernel:
            classifier = OneVsRestClassifier(SVC(kernel=kernel, C=C))
            cv = KFold(n_splits=10, shuffle=True)
    
            for train_index, test_index in cv.split(X):
                X_train, X_test, Y_train, Y_test = X[train_index], X[test_index], Y[train_index], Y[test_index] 
                classifier.fit(X_train, Y_train)
        
            scores = cross_val_score(classifier, X, Y, cv=10)
            best_acc = np.amax(scores)        
            candidate_acc.append(best_acc)
            print('For C = %d and kernel = %s, best 10 fold cross validation accuracy = %f' % (C, kernel,best_acc))
    
    return classifier

def PR_ovr_classifier(X, Y):
    print("Shape of X", X.shape)
    
    Y = label_binarize(Y, classes=[0, 1, 2])
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=1234)
    
    classifier = OneVsRestClassifier(LinearSVC(C=3, max_iter=10000))
    classifier.fit(X_train, Y_train)
    y_pred = classifier.predict(X_test)
    acc = accuracy_score(Y_test, y_pred)
    print('For value of C = 3, best 10 fold cross validation accuracy = %f' % (acc) )
    
    y_score = classifier.decision_function(X_test)
    n_classes = 3
    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(Y_test[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(Y_test[:, i], y_score[:, i])

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(Y_test.ravel(), y_score.ravel())
    average_precision["micro"] = average_precision_score(Y_test, y_score, average="micro")
    
    #print('Average precision score over all classes: {0:0.2f}'.format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Average precision score, over all classes: AP={0:0.2f}'.format(average_precision["micro"]))
    plt.show()
    
    # setup plot details
    colors = cycle(['red', 'yellow', 'green'])
    lines = []
    labels = []
    for i, color in zip(range(n_classes), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=2)
        lines.append(l)
        labels.append('Precision-recall for class {0} (area = {1:0.2f})'''.format(i, average_precision[i]))
    
    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Extension of Precision-Recall curve to multi-class')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=14))

    plt.show()

if __name__ == '__main__':
   
    file_name = 'bio_metadata.csv'
    df = pd.read_csv(file_name)

    # Filter metadata to retrieve only files desired
    filtered_df = getsplit.filter_df(df)

    X = filtered_df['language_num']
    Y = filtered_df['native_language']    
 
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    
    # Get resampled wav files using multiprocessing
    print('Loading wav files....')
    X = pool.map(get_wav, X)
    
    # Convert to features
    print('Feature Extraction...')
    X = pool.map(get_features, X)
    
    # Create segments from MFCCs
    X, Y = segments(X, Y)
    
    X = np.array(X)
    Y = np.array(Y)
    
    n_train_samples,n_train_x,n_train_y = X.shape
    
    X = X.reshape(n_train_samples,n_train_x*n_train_y)   
    
    classifierRF = rf_classifier(X, Y)
    classifierKNN = knn_classifier(X, Y)
    classifierOVR = ovr_classifier(X, Y)
    PR_ovr_classifier(X, Y)
