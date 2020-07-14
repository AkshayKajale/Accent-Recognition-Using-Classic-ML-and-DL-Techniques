# -*- coding: utf-8 -*-
import pandas as pd
from collections import Counter
from src import getsplit

import os
logdir = os.path.join('logs')

from keras import utils
import multiprocessing
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard
from keras.layers import LSTM, Masking
import accuracy

RATE = 24000
N_MFCC = 13
COL_SIZE = 30
EPOCHS = 150

def to_categorical(y):
    lang_dict = {}
    for index,language in enumerate(set(y)):
        lang_dict[language] = index
    y = list(map(lambda x: lang_dict[x],y))
    return utils.to_categorical(y, len(lang_dict))

def get_wav(language_num):
    y, sr = librosa.load('../audio/{}.wav'.format(language_num))
    return(librosa.core.resample(y=y,orig_sr=sr,target_sr=RATE, scale=True))

def mfcc_feature(wav):
    return(librosa.feature.mfcc(y=wav, sr=RATE, hop_length=int(RATE/40), n_fft=int(RATE/40), n_mfcc=N_MFCC))

def make_segments(mfccs,labels):
    segments = []
    seg_labels = []
    for mfcc,label in zip(mfccs,labels):
        for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
            segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
            seg_labels.append(label)
    return(segments, seg_labels)

def segment_one(mfcc):
    segments = []
    for start in range(0, int(mfcc.shape[1] / COL_SIZE)):
        segments.append(mfcc[:, start * COL_SIZE:(start + 1) * COL_SIZE])
    return(np.array(segments))

def create_segmented_mfccs(X_train):
    segmented_mfccs = []
    for mfcc in X_train:
        segmented_mfccs.append(segment_one(mfcc))
    return(segmented_mfccs)

def train_CNN(X_train, y_train, X_validation, y_validation, batch_size=128):
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])
    
    input_shape = (rows, cols, 1)
    
    X_train = X_train.reshape(X_train.shape[0], rows, cols, 1)
    
    X_validation = X_validation.reshape(X_validation.shape[0],val_rows,val_cols,1)


    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'training samples')
    
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3,3), activation='relu',
                     data_format="channels_last",
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64,kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
    
    tb = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    
    datagen = ImageDataGenerator(width_shift_range=0.05)
    
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(X_train) / 32
                        , epochs=EPOCHS, callbacks = [tb],
                        validation_data=(X_validation,y_validation))

    print("\n\nModel Summary:\n")
    model.summary()
    
    return model

def train_RNN(X_train, y_train, X_validation, y_validation, batch_size=128):
    rows = X_train[0].shape[0]
    cols = X_train[0].shape[1]
    val_rows = X_validation[0].shape[0]
    val_cols = X_validation[0].shape[1]
    num_classes = len(y_train[0])

    # input image dimensions to feed into 2D ConvNet Input layer
    input_shape = (rows, cols)
    #print(input_shape)
    X_train = X_train.reshape(X_train.shape[0], rows, cols)
    X_validation = X_validation.reshape(X_validation.shape[0], val_rows, val_cols)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'training samples')

    model = Sequential()
    model.add(LSTM(activation='tanh',units=64, recurrent_activation='sigmoid',
                   dropout=0.5, recurrent_dropout=0, unroll = False, use_bias = True,
                   return_sequences=True, stateful=False, input_shape=input_shape))
    
    model.add(LSTM(activation='tanh',units=64, recurrent_activation='sigmoid',
                   dropout=0.5, recurrent_dropout=0, unroll = False, use_bias = True,
                   return_sequences=True, stateful=False, input_shape=input_shape))
    
    model.add(LSTM(activation='tanh',units=64, recurrent_activation='sigmoid',
                   dropout=0.5, recurrent_dropout=0, unroll = False, use_bias = True,
                   return_sequences=True, stateful=False, input_shape=input_shape))
    
    model.add(Flatten())
    # Output layer
    model.add(Dense(num_classes, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    tb = TensorBoard(log_dir=logdir, histogram_freq=0, write_graph=True,
                     write_images=True, embeddings_freq=0, embeddings_layer_names=None,
                     embeddings_metadata=None)
    
    model.fit(X_train, y_train,
              batch_size=batch_size, epochs=EPOCHS,
              callbacks = [tb], validation_data=(X_validation,y_validation))

    print("\n\nModel Summary:\n")
    model.summary()
    
    return model


if __name__ == '__main__':
    
    file_name = 'bio_metadata.csv'
    
    # Load metadata
    df = pd.read_csv(file_name)
    
    filtered_df = getsplit.filter_df(df)
    
    X_train, X_test, y_train, y_test = getsplit.get_split(filtered_df)
    
    train_count = Counter(y_train)
    test_count = Counter(y_test)
    
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    print('Processing wav files....')
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    X_train = pool.map(get_wav, X_train)
    X_test = pool.map(get_wav, X_test)

    # Convert to MFCC
    print('Converting to MFCC....')
    X_train = pool.map(mfcc_feature, X_train)
    X_test = pool.map(mfcc_feature, X_test)

    # Create segments from MFCCs
    X_train, y_train = make_segments(X_train, y_train)
    X_validation, y_validation = make_segments(X_test, y_test)
    
    # Randomize training segments
    X_train, X_test_temp, y_train, y_test_temp = train_test_split(X_train, y_train, random_state=1234, test_size=0.01)
    X_train.extend(X_test_temp)
    y_train.extend(y_test_temp)
    
    RNNmodel = train_RNN(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation))
    CNNmodel = train_CNN(np.array(X_train), np.array(y_train), np.array(X_validation),np.array(y_validation))

    score, accuracy = RNNmodel.evaluate(
       np.array(X_validation), np.array(y_validation), batch_size=128, verbose=1)
    
    print('Test Accuracy of RNN model:', accuracy)
    
    y_predicted = accuracy.predict_class_all(create_segmented_mfccs(X_test), CNNmodel)

