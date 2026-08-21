# Copyright (C) 2021-2024 ETH Zurich
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
# ==============================================================================
#
# Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)


import os
import datetime
import json
import csv

import numpy as np
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def save_histogram(criterion, data, word):
    with open(criterion+"_"+word+'.csv', 'a', encoding="ISO-8859-1", newline='') as myfile:
        wr = csv.writer(myfile)
        wr.writerows(data)
    myfile.close()


def npy_to_txt(layer_number, activations):
    # Saving the input

    if layer_number == -1:
        tmp = activations.reshape(-1)
        f = open('input.txt', "a")
        f.write('# input (shape [1, 49, 10]),\\\n')
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()
    # Saving layers' activations
    else:
        tmp = activations.reshape(-1)
        f = open('out_layer' + str(layer_number) + '.txt', "a")
        f.write('layers.0.relu1 (shape [1, 25, 5, 64]),\\\n')  # Hardcoded, should be adapted for better understanding.
        for elem in tmp:
            if (elem < 0):
                f.write (str(256+elem) + ",\\\n")
            else:
                f.write (str(elem) + ",\\\n")
        f.close()


def remove_txt():
    # Removing old activations and inputs

    directory = '.'
    files_in_directory = os.listdir(directory)
    filtered_files = [file for file in files_in_directory if (file.startswith("out_layer") or file.startswith("input.txt"))]
    for file in filtered_files:
        path_to_file = os.path.join(directory, file)
        os.remove(path_to_file)


def conf_matrix(labels, predicted, training_parameters):
    # Plotting confusion matrix

    labels = labels.cpu()
    predicted = predicted.cpu()
    cm = confusion_matrix(labels, predicted)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(cm, index = [i for i in ['silence','unknown']+training_parameters['wanted_words']],
                  columns = [i for i in ['silence','unknown']+training_parameters['wanted_words']])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()


def per_noise_accuracy(labels, predicted, noises):

    noise_types = list(set(noises))

    for noise in noise_types:
        correct = 0
        total = 0

        for i in range (0, len(noises)):
            if (noises[i] == noise):
                total = total + 1
                if ((labels == predicted)[i]):
                    correct = correct + 1
        print('Noise number %3d - accuracy: %.3f' % (noise,  100 * correct / total))


def parameter_generation(args):

    # Opening JSON file
    with open(args['config_file']) as json_file:
        configuration = json.load(json_file)


    # Use manually passed parameters to overwrite the .json
    # Overwrite experimental parameters if exits manual definition
    for key, value in args.items():
        if value is not None:
            configuration['experimental_parameters'][key] = value
            
    # Overwrite training parameters if exits manual definition
    for key, value in args.items():
        if value is not None:
            configuration['training_parameters'][key] = value

    # Overwrite environment parameters if exits manual definition
    for key, value in args.items():
        if value is not None:
            configuration['environment_parameters'][key] = value

    # Overwrite environment parameters if exits manual definition
    for key, value in args.items():
        if value is not None:
            configuration['preprocessing_parameters'][key] = value

    # Preprocessing parameters
    configuration['preprocessing_parameters']['time_shift_samples'] = int((configuration['preprocessing_parameters']['time_shift_ms'] * configuration['preprocessing_parameters']['sample_rate']) / 1000)
    configuration['preprocessing_parameters']['desired_samples'] = int(configuration['preprocessing_parameters']['sample_rate'] * configuration['preprocessing_parameters']['clip_duration_ms'] / 1000)
    configuration['preprocessing_parameters']['window_size_samples'] = int(configuration['preprocessing_parameters']['sample_rate'] * configuration['preprocessing_parameters']['window_size_ms'] / 1000)
    configuration['preprocessing_parameters']['window_stride_samples'] = int(configuration['preprocessing_parameters']['sample_rate'] * configuration['preprocessing_parameters']['window_stride_ms'] / 1000)
    configuration['preprocessing_parameters']['length_minus_window']= (configuration['preprocessing_parameters']['desired_samples'] - configuration['preprocessing_parameters']['window_size_samples'])
    if configuration['preprocessing_parameters']['length_minus_window'] < 0:
        configuration['preprocessing_parameters']['spectrogram_length'] = 0
    else:
        configuration['preprocessing_parameters']['spectrogram_length'] = 1 + int(configuration['preprocessing_parameters']['length_minus_window'] / configuration['preprocessing_parameters']['window_stride_samples'])


    # Environment parameters  
    configuration['environment_parameters']['data_dir'] = configuration['environment_parameters']['data_dir_'+configuration['environment_parameters']['keywords_dataset']]

    # Training parameters
    if (configuration['training_parameters']['task'] == "gscv2_12w"):
        configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_12w']
    elif (configuration['training_parameters']['task'] == "gscv2_8w"):
        configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_8w']
    elif (configuration['training_parameters']['task'] == "gscv2_6w"):
        configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_6w']
    elif (configuration['training_parameters']['task'] == "gscv2_35w"):
        configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_35w']
    else:
        # Perform frequency-based selection for MSWC
        utterances_dict = {}
        with open(configuration['environment_parameters']['data_dir'][:-7]+"_align/en_splits.csv") as file:
          
            lines = file.readlines()
            for line in lines:
                # Count utterances per word - word frequency
                word = line.split(',')[1].split('/')[0]
                if word not in utterances_dict:
                    utterances_dict[word] = 0
                utterances_dict[word] = utterances_dict[word] + 1

        utterances_dict_ordered = sorted(utterances_dict.items(), key=lambda x: x[1], reverse=True)

        if (configuration['training_parameters']['wanted_frequency_mswc'] != 0):
            # Select files with more than "minimum_uttr" utterances
            minimum_uttr = configuration['training_parameters']['wanted_frequency_mswc']
            utterances_dict_select = [{'word': elem, 'count': utterances_dict[elem]} for elem in utterances_dict if utterances_dict[elem]>minimum_uttr]
            wanted_words = [elem['word'] for elem in utterances_dict_select]    

            configuration['training_parameters']['wanted_words'] = wanted_words
            
        else:
            # If no frequency was specified, use the GSC words for MSWC
            if ('12w' in configuration['training_parameters']['task']):
                configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_12w']
            elif ('8w' in configuration['training_parameters']['task']):
                configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_8w']
            elif ('6w' in configuration['training_parameters']['task']):
                configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_gscv2_6w']
            elif ('35w' in configuration['training_parameters']['task']):
                configuration['training_parameters']['wanted_words'] = configuration['training_parameters']['wanted_words_mswc_35w']
            else:
                print ("Unknown configuration.")

    configuration['training_parameters']['time_shift_samples'] = int((configuration['preprocessing_parameters']['time_shift_ms'] * configuration['preprocessing_parameters']['sample_rate']) / 1000)

    # Experimental parameters
    configuration['experimental_parameters']['date'] = str(datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    print ("------------ Complete configuration ------------")
    print (configuration)

    return configuration['environment_parameters'], configuration['preprocessing_parameters'], configuration['training_parameters'], configuration['experimental_parameters']
