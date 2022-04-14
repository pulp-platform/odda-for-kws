# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# Adapted by: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import hashlib
import math
import os.path
import random
import os
import re
import glob
import time
import torch
import torchaudio

from collections import Counter, OrderedDict
from pathlib import Path

import soundfile as sf
import numpy as np
import tensorflow as tf


MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_LABEL = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
  return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
  # Split dataset in training, validation, and testing set
  # Should be modified to load validation data from validation_list.txt
  # Should be modified to load testing data from testing_list.txt

  base_name = os.path.basename(filename)
  # We want to ignore anything after '_nohash_' in the file name when
  # deciding which set to put a wav in, so the data set creator has a way of
  # grouping wavs that are close variations of each other.
  hash_name = re.sub(r'_nohash_.*$', '', base_name)
  # This looks a bit magical, but we need to decide whether this file should
  # go into the training, testing, or validation sets, and we want to keep
  # existing files in the same set even if more files are subsequently
  # added.
  # To do that, we need a stable way of deciding based on just the file name
  # itself, so we do a hash of that and then use that to generate a
  # probability value that we use to assign it.
  hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
  percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
  if percentage_hash < validation_percentage:
    result = 'validation'
  elif percentage_hash < (testing_percentage + validation_percentage):
   result = 'testing'
  else:
    result = 'training'
  return result


class AudioProcessor(object):
  # Prepare data

  def __init__(self, training_parameters, data_processing_parameters):
      
      self.data_directory = training_parameters['data_dir']
      self.generate_background_noise(training_parameters)
      self.generate_data_dictionary(training_parameters)
      self.data_processing_parameters = data_processing_parameters


  def generate_data_dictionary(self, training_parameters):
    # For each data set, generate a dictionary containing the path to each file, its label, and its speaker.
    # Make sure the shuffling and picking of unknowns is deterministic.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    wanted_words_index = {}

    for index, wanted_word in enumerate(training_parameters['wanted_words']):
        wanted_words_index[wanted_word] = index + 2

    # Prepare data sets
    self.data_set = {'validation': [], 'testing': [], 'training': []}
    unknown_set = {'validation': [], 'testing': [], 'training': []}
    all_words = {}
    # Find all audio samples
    search_path = os.path.join(self.data_directory, '*', '*.wav')

    for wav_path in glob.glob(search_path):
      _ , word = os.path.split(os.path.dirname(wav_path))
      speaker_id = wav_path.split('/')[8].split('_')[0]  # Hardcoded, should use regex.
      word = word.lower()

      # Ignore background noise, as it has been handled by generate_background_noise()
      if word == BACKGROUND_NOISE_LABEL:
        continue

      all_words[word] = True
      # Determine the set to which the word should belong
      set_index = which_set(wav_path, training_parameters['validation_percentage'], training_parameters['testing_percentage'])

      # If it's a known class, store its detail, otherwise add it to the list
      # we'll use to train the unknown label.
      # If we use 35 classes - all are known, hence no unkown samples      
      if word in wanted_words_index:
        self.data_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})
      else:
        unknown_set[set_index].append({'label': word, 'file': wav_path, 'speaker': speaker_id})

    if not all_words:
      raise Exception('No .wavs found at ' + search_path)
    for index, wanted_word in enumerate(training_parameters['wanted_words']):
      if wanted_word not in all_words:
        raise Exception('Expected to find ' + wanted_word +
                        ' in labels but only found ' +
                        ', '.join(all_words.keys()))

    # We need an arbitrary file to load as the input for the silence samples.
    # It's multiplied by zero later, so the content doesn't matter.
    silence_wav_path = self.data_set['training'][0]['file']

    # Add silence and unknown words to each set
    for set_index in ['validation', 'testing', 'training']:

      set_size = len(self.data_set[set_index])
      silence_size = int(math.ceil(set_size * training_parameters['silence_percentage'] / 100))
      for _ in range(silence_size):
        self.data_set[set_index].append({
            'label': SILENCE_LABEL,
            'file': silence_wav_path,
            'speaker': "None" 
        })

      # Pick some unknowns to add to each partition of the data set.
      rand_unknown = random.Random(RANDOM_SEED)
      rand_unknown.shuffle(unknown_set[set_index])
      unknown_size = int(math.ceil(set_size * training_parameters['unknown_percentage'] / 100))
      self.data_set[set_index].extend(unknown_set[set_index][:unknown_size])

    # Make sure the ordering is random.
    for set_index in ['validation', 'testing', 'training']:
      rand_data_order = random.Random(RANDOM_SEED)
      rand_data_order.shuffle(self.data_set[set_index])

    # Prepare the rest of the result data structure.
    self.words_list = prepare_words_list(training_parameters['wanted_words'])
    self.word_to_index = {}
    for word in all_words:
      if word in wanted_words_index:
        self.word_to_index[word] = wanted_words_index[word]
      else:
        self.word_to_index[word] = UNKNOWN_WORD_INDEX
    self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX


  def generate_background_noise(self, training_parameters):
    # Load background noise, used to augment clean speech

    self.background_noise = []
    self.background_noise_name = []
    self.background_noise_test = []
    self.background_noise_test_name = []

    print (training_parameters)

    background_dir = os.path.join(training_parameters['noise_dir'])
    if not os.path.exists(background_dir):
      return self.background_noise

    for wav_path in sorted(Path(background_dir).rglob('*.wav')):
      # List of tensor, each one is a background noise
      sf_loader, _ = sf.read(wav_path)
      wav_file = torch.Tensor(np.array([sf_loader]))

      wav_path = str(wav_path)
      noise_type = wav_path.split('/')[6] #  TODO: index should be adapted based on the path

      if (training_parameters['noise_dataset'] == 'demand'):
        # Last 2 channels of test noises are used for testing
        # First 14 channels of the train noises are used for training
        if ([noise for noise in training_parameters['noise_test'] if noise in wav_path]):
          if ("ch15" in wav_path or "ch16" in wav_path):
            self.background_noise_test.append(wav_file[0])
            self.background_noise_test_name.append(noise_type)
        if ([noise for noise in training_parameters['noise_train'] if noise in wav_path]):
          if ("ch15" not in wav_path and "ch16" not in wav_path):
            self.background_noise.append(wav_file[0])
            self.background_noise_name.append(noise_type)
      else:
        self.background_noise.append(wav_file[0])
        self.background_noise_test.append(wav_file[0])
        self.background_noise_name.append(noise_type)
        self.background_noise_test_name.append(noise_type)

    if not self.background_noise:
      raise Exception('No background wav files were found in ' + search_path)
    if not self.background_noise_test:
      raise Exception('No test background wav files were found in ' + search_path)

  def get_size(self, mode):
    # Compute data set size

    return len(self.data_set[mode])


  def get_data(self, mode, training_parameters, task, offset):
    # Prepare and return data (utterances and labels) for inference

    # Pick one of the partitions to choose samples from
    candidates = self.data_set[mode]
    if training_parameters['batch_size'] == -1:
      samples_number = len(candidates)
    else:
      samples_number = max(0, min(training_parameters['batch_size'], len(candidates) - offset))

    # Create a data placeholder
    data_placeholder = np.zeros((samples_number, self.data_processing_parameters['spectrogram_length'],self.data_processing_parameters['feature_bin_count']),dtype='float32' )
    labels_placeholder = np.zeros(samples_number)
    noises_placeholder = np.zeros(samples_number)

    # Required for noise analysis
    if (training_parameters['noise_mode'] == 'nlkws'):
      use_background = (self.background_noise and (mode == 'training'))
    else:
      use_background = (self.background_noise and self.background_noise_test)

    if (task == -1):  # Silence task
      use_background = False

    pick_deterministically = (mode != 'training')

    for i in range(offset, offset + samples_number):

        # Pick which audio sample to use.
        if training_parameters['batch_size'] == -1 or pick_deterministically:
            # The randomness is eliminated here to train on the same batch ordering
            sample_index = i        
        else:
          # sample_index = np.random.randint(len(candidates))  # Random
          sample_index = i  # Ordered

        sample = candidates[sample_index]

        # Compute time shift offset
        if training_parameters['time_shift_samples'] > 0:
          time_shift_amount = np.random.randint(-training_parameters['time_shift_samples'], training_parameters['time_shift_samples'])
        else:
          time_shift_amount = 0
        if time_shift_amount > 0:
          time_shift_padding = [[time_shift_amount, 0], [0, 0]]
          time_shift_offset = [0, 0]
        else:
          time_shift_padding = [[0, -time_shift_amount], [0, 0]]
          time_shift_offset = [-time_shift_amount, 0]
        
        data_augmentation_parameters = {
            'wav_filename': sample['file'],
            'time_shift_padding': time_shift_padding,
            'time_shift_offset': time_shift_offset,
        }

        # Select background noise to mix in.
        if use_background:
          if (mode == 'training'):
            # Selecting the noises for training, where task=None assumes diverse noise-utterance pairs
            # whereas a non-None task implies the augmentation of all utterances with the same noise
            if (task is None):
              # background_index = np.random.randint(len(self.background_noise)) # Random augmentation
              background_index = i % len(self.background_noise) # Ordered augmentation
            else:
              background_index = np.random.randint(14*task, 14*(task+1))  
            background_samples = self.background_noise[background_index].numpy()

          else:
            # Selecting the noises for validation/testing
            if task is None:
              # background_index = np.random.randint(len(self.background_noise_test)) # Random augmentation
              background_index = i % len(self.background_noise_test) # Ordered augmentation  
            else:
              background_index = task
            background_samples = self.background_noise_test[background_index].numpy() 

          assert (len(background_samples) > self.data_processing_parameters['desired_samples'])

          background_offset = np.random.randint(0, len(background_samples) - self.data_processing_parameters['desired_samples'])
          background_clipped = background_samples[background_offset:(background_offset + self.data_processing_parameters['desired_samples'])]
          background_reshaped = background_clipped.reshape([self.data_processing_parameters['desired_samples'], 1])

          if sample['label'] == SILENCE_LABEL:
            if (training_parameters['noise_mode'] == 'odda'):
              background_volume = training_parameters['background_volume']  # Fixed volume
            else:              
              # background_volume = np.random.uniform(0, training_parameters['background_volume'])  # Variable volume
              background_volume = training_parameters['background_volume']  # Fixed volume
          elif np.random.uniform(0, 1) < training_parameters['background_frequency']:
            # background_volume = np.random.uniform(0, training_parameters['background_volume'])  # Variable volume
            background_volume = training_parameters['background_volume']  # Fixed volume
          else:
            background_volume = 0

        else:
          background_reshaped = np.zeros([self.data_processing_parameters['desired_samples'], 1])
          background_volume = 0
      
        data_augmentation_parameters['background_noise'] = background_reshaped
        data_augmentation_parameters['background_volume'] = background_volume

        # For silence samples, remove any sound
        if sample['label'] == SILENCE_LABEL:
          data_augmentation_parameters['foreground_volume'] = 0
        else:
          data_augmentation_parameters['foreground_volume'] = 1

        # Load data
        try:
            sf_loader, _ = sf.read(data_augmentation_parameters['wav_filename'])
            wav_file = torch.Tensor(np.array([sf_loader]))
        except:
            pass

        # Ensure data length is equal to the number of desired samples
        if len(wav_file[0]) < self.data_processing_parameters['desired_samples']:
            wav_file=torch.nn.ConstantPad1d((0,self.data_processing_parameters['desired_samples']-len(wav_file[0])),0)(wav_file[0])
        else:
            wav_file=wav_file[0][:self.data_processing_parameters['desired_samples']]
        scaled_foreground = torch.mul(wav_file, data_augmentation_parameters['foreground_volume'])

        # Padding wrt the time shift offset
        pad_tuple=tuple(data_augmentation_parameters['time_shift_padding'][0])
        padded_foreground = torch.nn.ConstantPad1d(pad_tuple,0)(scaled_foreground)
        sliced_foreground = padded_foreground[data_augmentation_parameters['time_shift_offset'][0]:data_augmentation_parameters['time_shift_offset'][0]+self.data_processing_parameters['desired_samples']]
        
        # Mix in background noise
        background_mul = torch.mul(torch.Tensor(data_augmentation_parameters['background_noise'][:,0]),data_augmentation_parameters['background_volume']) 
        background_add = torch.add(background_mul, sliced_foreground)

        # Compute MFCCs - PyTorch
        # melkwargs={ 'n_fft':1024, 'win_length':self.data_processing_parameters['window_size_samples'], 'hop_length':self.data_processing_parameters['window_stride_samples'],
        #        'f_min':20, 'f_max':4000, 'n_mels':40}
        # mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc=self.data_processing_parameters['feature_bin_count'], sample_rate=self.data_processing_parameters['desired_samples'], melkwargs=melkwargs, log_mels=True, norm='ortho')
        # data = mfcc_transformation(background_add)
        # data_placeholder[i - offset] = data[:,:self.data_processing_parameters['spectrogram_length']].numpy().transpose()

        # Compute MFCCs - TensorFlow (matching C-based implementation)
        tf_data = tf.convert_to_tensor(background_add.numpy(), dtype=tf.float32)
        tf_stfts = tf.signal.stft(tf_data, frame_length=self.data_processing_parameters['window_size_samples'], frame_step=self.data_processing_parameters['window_stride_samples'], fft_length=1024)
        tf_spectrograms = tf.abs(tf_stfts)
        power = True
        if power:
            tf_spectrograms = tf_spectrograms ** 2
        num_spectrogram_bins = tf_stfts.shape[-1]
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(40, num_spectrogram_bins, self.data_processing_parameters['desired_samples'], 20, 4000)
        tf_spectrograms = tf.cast(tf_spectrograms, tf.float32)
        tf_mel_spectrograms = tf.tensordot(tf_spectrograms, linear_to_mel_weight_matrix, 1)
        tf_mel_spectrograms.set_shape(tf_spectrograms.shape[:-1].concatenate(
                    linear_to_mel_weight_matrix.shape[-1:]))
        tf_log_mel = tf.math.log(tf_mel_spectrograms + 1e-6)
        tf_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf_log_mel)[..., :self.data_processing_parameters['feature_bin_count']]
        mfcc = torch.Tensor(tf_mfccs.numpy())
        data_placeholder[i - offset] = mfcc

        data_placeholder[i - offset] = np.clip(data_placeholder[i - offset] + 128, 0, 255)

        label_index = self.word_to_index[sample['label']]
        labels_placeholder[i - offset] = label_index

        if (task == -1):
          noises_placeholder[i - offset] = -1
        else:
          # Return the index of the noise from the complete noise list

          # Assumption: the test noise list contains all noises 
          # TODO: Search in the complete noise list
          if (mode == 'training'):
            noises_placeholder[i - offset] = self.background_noise_test_name.index(self.background_noise_name[background_index])
          else:
            noises_placeholder[i - offset] = self.background_noise_test_name.index(self.background_noise_test_name[background_index]) 

    return data_placeholder, labels_placeholder, noises_placeholder


class AudioGenerator(torch.utils.data.Dataset):
    # Returning batches of data (MFCCs) and labels

    def __init__(self, mode, audio_processor, training_parameters, task = None):
        self.mode = mode
        self.audio_processor = audio_processor
        if self.mode != 'training' and training_parameters['noise_mode'] == 'nlkws':
          training_parameters['background_frequency'] = 0
          training_parameters['background_volume'] = 0
          training_parameters['time_shift_samples'] = 0
        self.training_parameters = training_parameters

        # Preparing data for ODDA
        self.position = 0
        self.task = task


    def __len__(self):
        # Return dataset length

        if self.training_parameters['batch_size']==-1:
            return(len(self.audio_processor.data_set[self.mode]))
        else:
            return int(len(self.audio_processor.data_set[self.mode])/self.training_parameters['batch_size'])


    def __getitem__(self, idx):
        # Return a random batch of data, unless training_parameters['batch_size'] == -1

        data, labels, noises = self.audio_processor.get_data(self.mode, self.training_parameters, self.task, self.position)        

        self.position += self.training_parameters['batch_size']
        
        return data, labels, noises
