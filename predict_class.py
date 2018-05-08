#! /usr/bin/env python3
'''
Given one audio clip, output what the network thinks.
Taken mostly from drscotthawley from panotti, with changes for integration.
'''
from __future__ import print_function
import numpy as np
import librosa
import os
from os.path import isfile
from panotti.models import *
from panotti.datautils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less TF messages

def get_canonical_shape(signal):
    """Finds the shape of a given audio signal loadded in librosa.
    Returns the shape as a tuple of ints.
    """
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape


def predict_one(signal, sr, model, expected_melgram_shape):# class_names, model)#, weights_file="weights.hdf5"):
    """Predicts for one audio file. The name is leftover from the panotti classifier,
    where the function could be used to predict a list of files. Returns a numpy
    array of confidence percentages.
    """
    X = make_layered_melgram(signal,sr)
    print("signal.shape, melgram_shape, sr = ",signal.shape, X.shape, sr)

    if (X.shape[1:] != expected_melgram_shape):   # resize if necessary, pad with zeros
        Xnew = np.zeros([1]+list(expected_melgram_shape))
        min1 = min(  Xnew.shape[1], X.shape[1]  )
        min2 = min(  Xnew.shape[2], X.shape[2]  )
        min3 = min(  Xnew.shape[3], X.shape[3]  )
        Xnew[0,:min1,:min2,:min3] = X[0,:min1,:min2,:min3]  # truncate
        X = Xnew
    return model.predict(X,batch_size=1,verbose=False)[0]


def main_given_filename(filename):
    """Alteration of the panotti main function, altered to take a filename
    instead of system arguments. Given a filename, returns the most likely class
    as a string. Also prints confidence for all classes in terminal.
    """
    np.random.seed(1)
    #I am gonna hardcode these for now, sorry, they were default values for sys args
    weights_file= 'weights.hdf5'
    dur = None
    resample = 44100
    mono = False

    # Load the model
    model, class_names = load_model_ext(weights_file)
    if model is None:
        print("No weights file found.  Aborting")
        exit(1)

    nb_classes = len(class_names)
    print(nb_classes," classes to choose from: ",class_names)
    expected_melgram_shape = model.layers[0].input_shape[1:]
    print("Expected_melgram_shape = ",expected_melgram_shape)
    file_count = 0

    if os.path.isfile(filename):
        file_count += 1
        print("File",filename,":",end="")

        signal, sr = load_audio(filename, mono=mono, sr=resample)#sr is sample rate

        y_prob = predict_one(signal, sr, model, expected_melgram_shape)#Probabilities for each class (Confidence values)

        for i in range(nb_classes):
            print( class_names[i],": ",y_prob[i],", ",end="",sep="")#Prints confidence values for all classes
        answer = class_names[ np.argmax(y_prob)]#Answer is the one with the highest confidence
        print("--> ANSWER:", class_names[ np.argmax(y_prob)])
        return answer
    else:
        #If the file is not a file, say so.
        print(" *** File",filename,"does not exist.  Skipping.")
        return "Try again, please"
