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


import glob
import hashlib
import math
import os
import random
import re
import torch

import numpy as np
import soundfile as sf

from collections import Counter, OrderedDict
from copy import deepcopy
from pathlib import Path

class DatasetCreator(object):
    
    # Prepare data
    def __init__(self, environment_parameters, training_parameters, preprocessing_parameters):
            self.random_seed = 59185 


            # Set environment variables
            self.max_num_wavs_per_class = 2**27 - 1    # ~134M
            self.background_noise_label = '_background_noise_'
            self.silence_label = '_silence_'
            self.silence_index = 0
            self.unknown_word_label = '_unknown_'
            self.unknown_word_index = 1
            

            self.environment_parameters = environment_parameters
            self.training_parameters = training_parameters
            self.preprocessing_parameters = preprocessing_parameters

            self.generate_background_noise()
            if (self.training_parameters['reverb'] == "true"):
                self.generate_reverberant_rooms()

            self.prepare_words_list()

            self.generate_data_dictionary()

            self.curate_words_list()


    def curate_words_list(self):

        self.word_to_index = {}
        for word in self.all_words:
            if word in self.wanted_words_index:
                self.word_to_index[word] = self.wanted_words_index[word]
            else:
                self.word_to_index[word] = self.unknown_word_index
        if (len(self.words_list) == 12 and self.environment_parameters['keywords_dataset'] == "gscv2"):
            self.word_to_index[self.silence_label] = self.silence_index 
        elif (len(self.words_list) == 8 and self.environment_parameters['keywords_dataset'] == "gscv2"):
            self.word_to_index[self.silence_label] = self.silence_index 
        # Patch
        if (self.environment_parameters['keywords_dataset'] == "kinem"):
            self.word_to_index[self.silence_label] = self.silence_index 


    def prepare_words_list(self):

        # self.words_list and self.wanted_words_index can be combined

        if (self.environment_parameters['keywords_dataset'] == "gscv2"):
            if (len(self.training_parameters['wanted_words']) == 10):
                self.words_list = [self.silence_label, self.unknown_word_label] + self.training_parameters['wanted_words']    # 12 words 
            else:
                self.words_list = self.training_parameters['wanted_words']    # 6 / 35 words
        elif (self.environment_parameters['keywords_dataset'] == "mswc"):
            self.words_list = self.training_parameters['wanted_words']
        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            if (len(self.training_parameters['wanted_words']) == 10):
                self.words_list = [self.silence_label, self.unknown_word_label] + self.training_parameters['wanted_words']    # 12 words 
            else:
                self.words_list = self.training_parameters['wanted_words']

        self.wanted_words_index = {}
        for index, wanted_word in enumerate(self.training_parameters['wanted_words']):
            if (self.training_parameters['task'] =="gscv2_12w" or self.training_parameters['task'] =="gscv2_8w"):
                self.wanted_words_index[wanted_word] = index + 2    # 12 words
            else:
                self.wanted_words_index[wanted_word] = index    # 6 / 35 words / MSWC


    # Split dataset in training, validation, and testing set
    def which_set(self, filename, validation_percentage, testing_percentage, dataset_path):

        if (self.environment_parameters['keywords_dataset'] == 'gscv2'):
            # Consider loading from validation_list.txt
            # Consider loading from testing_list.txt

            base_name = os.path.basename(filename)
            hash_name = re.sub(r'_nohash_.*$', '', base_name)
            hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                                    (self.max_num_wavs_per_class + 1)) *
                                                 (100.0 / self.max_num_wavs_per_class))
            if percentage_hash < validation_percentage:
                result = 'validation'
            elif percentage_hash < (testing_percentage + validation_percentage):
             result = 'testing'
            else:
                result = 'training'
            return result

        elif (self.environment_parameters['keywords_dataset'] == 'mswc'):
            if filename in open(dataset_path[:-7]+"_align/en_dev.csv").read():
                result = 'validation'
            elif filename in open(dataset_path[:-7]+"_align/en_test.csv").read():
                result = 'testing'
            else:
                result = 'training'
            return result

        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            raise Exception ("Dataset split already defined in dataset directory.")
        else:
            raise ValueError ("Dataset not defined.")


    # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
    # Ensure deterministic data shuffling
    def generate_data_dictionary(self):

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

        # Creating occurences dictionary for frequency-based selection (e.g., MSWC datset)
        occurences_dict = {'testing': {}, 'validation': {}, 'training': {}}
        for word in self.training_parameters['wanted_words']:
            occurences_dict['testing'][word] = 0
            occurences_dict['validation'][word] = 0
            occurences_dict['training'][word] = 0

        # Prepare data sets
        self.all_words = {}
        self.data_set = {'validation': [], 'testing': [], 'training': [], 'odda': [], 'odda_val': []}
        unknown_set = {'validation': [], 'testing': [], 'training': [], 'odda': [], 'odda_val': []}
        
        # Find all audio samples
        if (self.environment_parameters['keywords_dataset'] == "gscv2"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*.wav')
        elif(self.environment_parameters['keywords_dataset'] == "mswc"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*.wav')
        elif(self.environment_parameters['keywords_dataset'] == "kinem"):
            search_path = os.path.join(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']], '*', '*', '*.wav.wav')

        if (self.environment_parameters['keywords_dataset'] == "gscv2"):  # Parsing the files individually
            for wav_path in glob.glob(search_path):

                _ , word = os.path.split(os.path.dirname(wav_path))
                word = word.lower()
                speaker_id = wav_path.split('/')[-1].split('_')[0]  # Hardcoded, should use regex.

                # Ignore background noise, as it has been handled by generate_background_noise()
                if word == self.background_noise_label:
                    continue

                self.all_words[word] = True
                # Determine the set to which the word should belong
                set_index = self.which_set(wav_path, self.training_parameters['validation_percentage'], \
                    self.training_parameters['testing_percentage'], self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']])

                # If it's a known class, store its detail, otherwise add it to 'unknown'
                # For 35 target classes, there are no 'unknown samples'
                if word in self.wanted_words_index:
                    self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                else:
                    unknown_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})

        elif (self.environment_parameters['keywords_dataset'] == "mswc"): # Parsing the lists organizing the files
            with open(self.environment_parameters['data_dir'][:-7]+"_align/en_dev.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['validation'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['validation'][word] += 1

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']][:-7]+"_align/en_test.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['testing'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['testing'][word] += 1

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']][:-7]+"_align/en_train.csv", newline='') as csvfile:
                lines = csv.reader(csvfile, delimiter=',', quotechar='|')
                for row in lines:
                    word = row[1]
                    if (word in self.wanted_words_index):

                        self.all_words[word] = True
                        wav_path = str(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+row[0])+".wav"
                        speaker_id = row[0].split('_')[3].split('.')[0]
                        self.data_set['training'].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
                        # Add word in occurences dict
                        occurences_dict['training'][word] += 1

        elif (self.environment_parameters['keywords_dataset'] == "kinem"):
            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['target_noise']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_test.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['testing'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['testing'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['target_noise']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_train.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['training'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['training'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

            # KINEM has no validation dataset, so we use the training data naively for validation
            with open(self.environment_parameters['data_dir_'+self.environment_parameters['keywords_dataset']]+'/noise'+self.environment_parameters['target_noise']+'_distance'+ \
                self.environment_parameters['distance'] + "_microphone" + self.environment_parameters['microphone'] + '_train.txt') as txtfile:

                lines = [line for line in txtfile]
                for line in lines:
                    class_index = line.split('/')[-1]
                    match = re.search(r'([a-zA-Z]+)(\d+)\.wav\.wav', class_index)
                    if match:
                        class_c, index_i = match.groups()
                    else:
                        # Handle the case where the regex doesn't match
                        class_c, index_i = None, None
                    speaker_id = line.split('/')[-2].split('_')[0].split('S')[-1]
                    if word in self.wanted_words_index:
                        self.data_set['validation'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    else:
                        unknown_set['validation'].append({'label': class_c.lower(), 'file': line.strip(), 'speaker': speaker_id})
                    self.all_words[class_c.lower()] = True

        if not self.all_words:
            raise Exception('No .wavs found at ' + search_path)

        for index, wanted_word in enumerate(self.training_parameters['wanted_words']):
            if wanted_word not in self.all_words:
                if (self.environment_parameters['keywords_dataset'] == 'kinem'):
                    continue    # data could miss in evaluation
                raise Exception('Expected to find ' + wanted_word +
                                                ' in labels but only found ' +
                                                ', '.join(self.all_words.keys()))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_set['training'][0]['file']

        # Add silence and unknown words to each set
        for set_index in ['validation', 'testing', 'training']:

            set_size = len(self.data_set[set_index])
            silence_size = int(math.ceil(set_size * self.training_parameters['silence_percentage'] / 100))
            for _ in range(silence_size):
                self.data_set[set_index].append({
                        'label': self.silence_label,
                        'file': silence_wav_path,
                        'speaker': "None" 
                })

            # Pick some unknowns to add to each partition of the data set.
            rand_unknown = random.Random(self.random_seed)
            rand_unknown.shuffle(unknown_set[set_index])
            unknown_size = int(math.ceil(set_size * self.training_parameters['unknown_percentage'] / 100))
            self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])


        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            rand_data_order = random.Random(self.random_seed)
            rand_data_order.shuffle(self.data_set[set_index])

        # Initializing ODDA data as the training data
        # Required for pretraining, followed by selection and adaptation
        self.data_set['odda'] = deepcopy(self.data_set['training'])
        self.data_set['odda_val'] = deepcopy(self.data_set['validation'])


    # Load noise dataset
    def generate_noise_set(self, dataset_directory, dataset_name, noise_list, online, test):

        recordings = []
        names = []

        if (online):
            if (test):
                noise_key = "online_noise_test_dataset"
            else:
                noise_key = "online_noise_train_dataset"
        else:
            if (test):
                noise_key = "offline_noise_test_dataset"
            else:
                noise_key = "offline_noise_train_dataset"


        if (dataset_name == "kinem"):
            background_dir = os.path.join(self.environment_parameters['noise_dir_'+self.environment_parameters[noise_key]])
            background_dir = glob.glob(background_dir+'/' + noise_list[0] +'/S*_' + self.environment_parameters['microphone'] + self.environment_parameters['distance'])[0]
        else:
            background_dir = os.path.join(self.environment_parameters['noise_dir_'+self.environment_parameters[noise_key]])


        if not os.path.exists(background_dir):
            raise OSError("Background noise directory not found.")

        # Iterate through existing .wavs
        for wav_path in sorted(Path(background_dir).rglob('*.wav')):

            if (dataset_name == "gscv2"):
                noise_type = str(wav_path).split('/')[-1].split('.wav')[0]
            elif (dataset_name == "demand"):
                noise_type = str(wav_path).split('/')[-2]
            elif (dataset_name == "kinem"):
                noise_type = str(wav_path).split('/')[-3]
            else:
                raise ValueError("Dataset management not defined.")

            if noise_type in noise_list:
                if (dataset_name == "demand"):
                    if (test is True):
                        if ("ch15" in str(wav_path) or "ch16" in str(wav_path)):

                            noise_path = str(wav_path)
                            sf_loader_noise, _ = sf.read(noise_path)
                            wav_background_samples = torch.from_numpy(sf_loader_noise).float()

                            recordings.append(wav_background_samples)
                            names.append(noise_type)
                    else:
                        if ("ch15" not in str(wav_path) and "ch16" not in str(wav_path)):
                            noise_path = str(wav_path)
                            sf_loader_noise, _ = sf.read(noise_path)
                            wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                            
                            recordings.append(wav_background_samples)
                            names.append(noise_type)

                elif (dataset_name == "gscv2"):
                    noise_path = str(wav_path)
                    sf_loader_noise, _ = sf.read(noise_path)
                    wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                    
                    recordings.append(wav_background_samples)
                    names.append(noise_type)

                elif (dataset_name == "kinem"):
                    if ("background_noise.wav.wav" in str(wav_path)):
                        noise_path = str(wav_path)

                        sf_loader_noise, _ = sf.read(noise_path)
                        wav_background_samples = torch.from_numpy(sf_loader_noise).float()
                        
                        recordings.append(wav_background_samples[:, 0]) # add all samples of the first channel
                        names.append(noise_type)


        if ("SILENCE" in noise_list):
            # Add as many silence samples as samples for other noises
            n_samples = int(len(recordings)/(len(noise_list)-1))

            for i in range (0, n_samples):
                recordings.append(torch.Tensor(np.zeros(20000)))
                names.append("SILENCE")

        if not recordings:
            raise Exception('No background wav files were found in ' + dataset_directory)

        return recordings, names


    # Load complete set of background noises
    def generate_background_noise(self):
        

        self.offline_background_noise_train, self.offline_background_noise_train_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["offline_noise_train_dataset"]],
            self.environment_parameters["offline_noise_train_dataset"],
            self.environment_parameters["offline_noise_train"],
            "offline",
            "train"
            )
        self.offline_background_noise_test, self.offline_background_noise_test_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["offline_noise_test_dataset"]],
            self.environment_parameters["offline_noise_test_dataset"],
            self.environment_parameters["offline_noise_test"],
            "offline",
            "test"
            )
        self.online_background_noise_train, self.online_background_noise_train_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["online_noise_train_dataset"]],
            self.environment_parameters["online_noise_train_dataset"],
            self.environment_parameters["online_noise_train"],
            "online",
            "train"
            )
        self.online_background_noise_test, self.online_background_noise_test_name = self.generate_noise_set(\
            self.environment_parameters["noise_dir_"+self.environment_parameters["online_noise_test_dataset"]],
            self.environment_parameters["online_noise_test_dataset"],
            self.environment_parameters["online_noise_test"],
            "online",
            "test"
            )


    # Generate reverb model
    def generate_reverberant_rooms(self):

        # For room parameters, see wham_room.py
        n_room_train = self.environment_parameters['reverb_train_n']
        n_room_val = self.environment_parameters['reverb_val_n']
        n_room_test = self.environment_parameters['reverb_test_n']

        modes = ["training", "validation", "testing"] 

        n_room = {
        "training": self.environment_parameters['reverb_train_n'],
        "validation": self.environment_parameters['reverb_val_n'],
        "testing": self.environment_parameters['reverb_test_n']
        }

        self.reverb_rooms = {
        "training": [],
        "validation": [],
        "testing": []
        }

        self.anechoic_rooms = {
        "training": [],
        "validation": [],
        "testing": []
        }

        for mode in modes:
            for idx_room in range(n_room[mode]):
                room_param_dict = reverb.gen_room_params()
                self.anechoic_rooms[mode].append(reverb.gen_anechoic_room(room_param_dict))
                self.reverb_rooms[mode].append(reverb.gen_reverb_room(room_param_dict))


    # Compute data set size
    def get_size(self, mode):
        
        return len(self.data_set[mode])