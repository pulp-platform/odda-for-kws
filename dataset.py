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


import hashlib
import math
import os.path
import random
import os
import re
import glob
import time
import torch
import csv

from collections import Counter, OrderedDict
from pathlib import Path
from copy import deepcopy
import reverb

import soundfile as sf
import numpy as np


class DatasetProcessor(torch.utils.data.Dataset):
        

    def __init__(self, mode, dataset_creator, training_parameters, device = None, task = None):

        self.mode = mode
        self.task = task
        self.device = device

        self.dataset_creator = dataset_creator

        if self.mode != 'training' and training_parameters['noise_mode'] == 'noiseless':
            training_parameters['background_frequency'] = 0
            training_parameters['background_volume'] = 0
            training_parameters['time_shift_samples'] = 0

        self.training_parameters = training_parameters
        # TODO: Create getter and setter methods
        self.environment_parameters = self.dataset_creator.environment_parameters
        self.preprocessing_parameters = self.dataset_creator.preprocessing_parameters
        self.word_to_index = self.dataset_creator.word_to_index
        self.data_set = self.dataset_creator.data_set

        self.offline_background_noise_train = self.dataset_creator.offline_background_noise_train
        self.offline_background_noise_test = self.dataset_creator.offline_background_noise_test
        self.online_background_noise_train = self.dataset_creator.online_background_noise_train
        self.online_background_noise_test = self.dataset_creator.online_background_noise_test

        self.offline_background_noise_train_name = self.dataset_creator.offline_background_noise_train_name
        self.offline_background_noise_test_name = self.dataset_creator.offline_background_noise_test_name
        self.online_background_noise_train_name = self.dataset_creator.online_background_noise_train_name
        self.online_background_noise_test_name = self.dataset_creator.online_background_noise_test_name

        self.offline_noise_train_dataset = self.dataset_creator.environment_parameters["offline_noise_train_dataset"]
        self.offline_noise_test_dataset = self.dataset_creator.environment_parameters["offline_noise_test_dataset"]
        self.online_noise_test_dataset = self.dataset_creator.environment_parameters["online_noise_test_dataset"]
        self.online_noise_train_dataset = self.dataset_creator.environment_parameters["online_noise_train_dataset"]


        self.data_augmentation_parameters = {}

        self.default_random_seed = self.dataset_creator.random_seed

        if (self.mode == "testing"):
            random.seed(self.default_random_seed)
            np.random.seed(self.default_random_seed)
        else:
            random.seed(self.preprocessing_parameters['noise_seed'])
            np.random.seed(self.preprocessing_parameters['noise_seed'])

    def __len__(self):

        # Return dataset length
        return len(self.dataset_creator.data_set[self.mode])


    def setup(self, offset):

        # Select sample
        pick_deterministically = (self.mode != 'training')

        # Pick which audio sample to use.
        if self.training_parameters['batch_size'] == -1 or pick_deterministically:
            # The randomness is eliminated here to train on the same batch ordering
            sample_index = offset                
        else:
            # TODO: Integrate argument to decide on training order
            # sample_index = np.random.randint(len(candidates))    # Random
            sample_index = offset    # Ordered
        
        self.sample = self.data_set[self.mode][sample_index]


        # TODO: use_background should be true for all training modes
        if (self.training_parameters['noise_mode'] == 'noiseless'):
            self.use_background = False
        else:
            self.use_background = (len(self.offline_background_noise_train) and len(self.offline_background_noise_test)) or \
            (len(self.online_background_noise_train) and len(self.online_background_noise_test))

        # TODO: Obsolete, remove
        if (self.task == -1):    # Silence task
            self.use_background = False

        # Select background noise list and volume
        if self.use_background:
            if (self.mode == 'training'):
                # task=None assumes diverse noise-utterance pairs
                # task=idx assumes specific noise-utterance pairs
                if (self.task is None):
                    # TODO: Integrate argument to decide on training order
                    # background_index = np.random.randint(len(self.offline_background_noise_train)) # Random augmentation
                    background_index = offset % len(self.offline_background_noise_train) # Ordered augmentation
                else:
                    background_index = np.random.randint(14*self.task, 14*(self.task+1))  
                wav_background_samples = self.offline_background_noise_train[background_index]

            elif (self.mode == 'validation' or self.mode == 'testing'):
                # Selecting the noises for validation/testing
                if self.task is None:
                    # TODO: Integrate argument to decide on training order
                    # background_index = np.random.randint(len(self.offline_background_noise_test)) # Random augmentation
                    background_index = offset % len(self.offline_background_noise_test) # Ordered augmentation    
                else:
                    background_index = self.task
                wav_background_samples = self.offline_background_noise_test[background_index] 

            elif (self.mode == 'odda'):
                # task=None assumes diverse noise-utterance pairs
                # task=idx assumes specific noise-utterance pairs
                if (self.task is None):
                    # TODO: Integrate argument to decide on training order
                    # background_index = np.random.randint(len(self.online_background_noise_train)) # Random augmentation
                    background_index = offset % len(self.online_background_noise_train) # Ordered augmentation
                else:
                    background_index = np.random.randint(14*self.task, 14*(self.task+1))   
                wav_background_samples = self.online_background_noise_train[background_index]

            elif (self.mode == 'odda_val'):
                # Selecting the noises for validation/testing
                if self.task is None:
                    # TODO: Integrate argument to decide on training order
                    # background_index = np.random.randint(len(self.online_background_noise_test)) # Random augmentation
                    background_index = offset % len(self.online_background_noise_test) # Ordered augmentation    
                else:
                    background_index = self.task
                wav_background_samples = self.online_background_noise_test[background_index] 

            assert (len(wav_background_samples) > self.preprocessing_parameters['desired_samples'])            

            background_offset = np.random.randint(0, len(wav_background_samples) - self.preprocessing_parameters['desired_samples'])
            background_clipped = wav_background_samples[background_offset:(background_offset + self.preprocessing_parameters['desired_samples'])]
            background_reshaped = torch.reshape(background_clipped, (self.preprocessing_parameters['desired_samples'], 1))

            if self.sample['label'] == '_silence_':
                background_volume = self.training_parameters['background_volume'] # Fixed volume
            elif np.random.uniform(0, 1) < self.training_parameters['background_frequency']:
                # TODO: Parametrize variable vs fixed volume selection
                # background_volume = np.random.uniform(0, self.training_parameters['background_volume']) # Variable volume
                background_volume = self.training_parameters['background_volume'] # Fixed volume
            else:
                background_volume = 0

        else:
            background_reshaped = torch.from_numpy(np.zeros([self.preprocessing_parameters['desired_samples'], 1]))
            background_volume = 0
            background_index = -1 # TODO: Decide on value for noiseless
    
        self.data_augmentation_parameters['background_noise'] = background_reshaped
        self.data_augmentation_parameters['background_volume'] = background_volume
        self.data_augmentation_parameters['background_index'] = background_index

        # For silence samples, remove any sound
        if self.sample['label'] == '_silence_':
            self.data_augmentation_parameters['foreground_volume'] = 0
        else:
            self.data_augmentation_parameters['foreground_volume'] = 1


        # Compute time shift offset
        if self.training_parameters['time_shift_samples'] > 0:
            time_shift_amount = np.random.randint(-self.training_parameters['time_shift_samples'], self.training_parameters['time_shift_samples'])
        else:
            time_shift_amount = 0
        if time_shift_amount > 0:
            time_shift_padding = [[time_shift_amount, 0], [0, 0]]
            time_shift_offset = [0, 0]
        else:
            time_shift_padding = [[0, -time_shift_amount], [0, 0]]
            time_shift_offset = [-time_shift_amount, 0]
        
        self.data_augmentation_parameters['sample_filename'] = self.sample['file']
        self.data_augmentation_parameters['time_shift_padding'] = time_shift_padding
        self.data_augmentation_parameters['time_shift_offset'] = time_shift_offset

        # Add reverb room
        if (self.training_parameters['reverb'] == "true"):
            if (self.training_parameters["reverb_mode"] == "training_only" and (self.mode == "training" or self.mode == "odda")):
                # CPU - preexisting rooms
                room_idx = offset % len(self.reverb_rooms["training"])
                reverb_room = self.reverb_rooms["training"][room_idx]
                anechoic_room = self.anechoic_rooms["training"][room_idx]

            elif (self.training_parameters["reverb_mode"] == "training_all"):
                # CPU - preexisting rooms
                room_idx = offset % len(self.reverb_rooms["training"])
                reverb_room = self.reverb_rooms["training"][room_idx]
                anechoic_room = self.anechoic_rooms["training"][room_idx]

            elif (self.training_parameters["reverb_mode"] == "training_each"):
                # CPU - preexisting rooms
                if (self.mode == 'odda'):
                    mmode = 'training'
                elif (self.mode == 'odda_val'):
                    mmode = 'validation'
                else:
                    mmode = self.mode
                room_idx = offset % len(self.reverb_rooms[mmode])
                reverb_room = self.reverb_rooms[mmode][room_idx]
                anechoic_room = self.anechoic_rooms[mmode][room_idx]

            elif (self.training_parameters["reverb_mode"] == "eval_only" and (self.mode == 'validation' or self.mode == 'testing' or self.mode == 'odda_val')):
                # CPU - preexisting rooms
                if (self.mode == 'odda_val'):
                    mmode = 'validation'
                else:
                    mmode = self.mode
                room_idx = offset % len(self.reverb_rooms[mmode])
                reverb_room = self.reverb_rooms[mmode][room_idx]
                anechoic_room = self.anechoic_rooms[mmode][room_idx]

            self.data_augmentation_parameters['reverb_room'] = reverb_room
            self.data_augmentation_parameters['anechoic_room'] = anechoic_room


    def augment(self):

        # Load data
        try:
            if (self.environment_parameters['keywords_dataset'] == 'gscv2'):
                sf_loader, _ = sf.read(self.data_augmentation_parameters['sample_filename']) 
            elif (self.environment_parameters['keywords_dataset'] == 'mswc'):
                if (os.path.isfile(self.data_augmentation_parameters['sample_filename'])):
                    if self.data_augmentation_parameters['sample_filename'].endswith('opus') :
                        self.data_augmentation_parameters['sample_filename'] = self.data_augmentation_parameters['sample_filename'].replace("opus", "wav")
                    sf_loader, _ = sf.read(self.data_augmentation_parameters['sample_filename'])
                else:
                    opus_path = self.data_augmentation_parameters['sample_filename'].replace("wav", "opus")
                    command = f"ffmpeg -i '{opus_path}' -ac 1    pcm_s24le '{self.data_augmentation_parameters['sample_filename']}'"
                    subprocess.call(command,shell=True)
                    sf_loader, _ = sf.read(self.data_augmentation_parameters['sample_filename'])

            elif (self.environment_parameters['keywords_dataset'] == "kinem"):
                sf_loader, _ = sf.read(self.data_augmentation_parameters['sample_filename'])     

            wav_file = torch.from_numpy(sf_loader).float().to(self.device)

        except:
            raise Exception("Could not load the sample.")

        # Adding the path as a returned variable
        self.paths_placeholder = self.data_augmentation_parameters['sample_filename']

        # Ensure data length is equal to the number of desired samples
        if len(wav_file) < self.preprocessing_parameters['desired_samples']:
            wav_file=torch.nn.ConstantPad1d((0,self.preprocessing_parameters['desired_samples']-len(wav_file)),0)(wav_file)
        else:
            wav_file=wav_file[:self.preprocessing_parameters['desired_samples']]

        scaled_foreground = torch.mul(wav_file, self.data_augmentation_parameters['foreground_volume'])

        # Padding wrt the time shift offset
        pad_tuple=tuple(self.data_augmentation_parameters['time_shift_padding'][0])
        padded_foreground = torch.nn.ConstantPad1d(pad_tuple,0)(scaled_foreground)
        sliced_foreground = padded_foreground[self.data_augmentation_parameters['time_shift_offset'][0]:self.data_augmentation_parameters['time_shift_offset'][0]+self.preprocessing_parameters['desired_samples']]

        # Mix in background noise        
        background_mul = torch.mul(self.data_augmentation_parameters['background_noise'],self.data_augmentation_parameters['background_volume']).to(self.device)

        # Compute SNR
        sliced_foreground_energy = sliced_foreground**2
        background_mul_energy = background_mul**2

        avg_foreground_power = torch.mean(sliced_foreground_energy, dtype=torch.float64)
        avg_backgroung_power = torch.mean(background_mul_energy, dtype=torch.float64)

        SNR = 10 * torch.log10(avg_foreground_power/avg_backgroung_power+1e-6)
        sum_background_mul_energy = torch.sum(background_mul_energy, dtype=torch.float64)
        sum_sliced_foreground_energy = torch.sum(sliced_foreground_energy, dtype=torch.float64)

        # TODO: Revert to single SNR @ test/evaluation
        if (self.mode == 'training' or self.mode == 'odda'):
            if (len(self.training_parameters['snr_range']) > 1):
                curr_snr = np.random.uniform(self.training_parameters['snr_range'][0], self.training_parameters['snr_range'][1]) 
            else:
                curr_snr = self.training_parameters['snr_range'][0]
        else:
            # TODO: Parametrize single vs interval SNR selection for test/evaluation
            if (len(self.training_parameters['snr_range']) > 1):
                curr_snr = np.random.uniform(self.training_parameters['snr_range'][0], self.training_parameters['snr_range'][1]) 
            else:
                curr_snr = self.training_parameters['snr_range'][0]

        k = torch.sqrt( ((sum_background_mul_energy**curr_snr)/(sum_sliced_foreground_energy**(curr_snr-SNR)))**(1/SNR) / sum_background_mul_energy )
    
        # Add reverb
        if (self.training_parameters['reverb'] == "true" and ((self.mode == "odda") or (self.mode == "odda_val") or (self.mode == "testing") )):
            sliced_foreground_reverb = reverb.gen_signal(self.data_augmentation_parameters['reverb_room'], self.data_augmentation_parameters['anechoic_room'],
                signal=sliced_foreground.cpu().numpy(), fs=self.preprocessing_parameters['desired_samples'])
            sliced_foreground = torch.tensor(sliced_foreground_reverb).cuda()

        if (self.use_background):
            # NOTE: Is this the best way to bypass SNR-based scaling?
            # NOTE: The comparison yields False because snr_range is a list. 
            # All the experiments on kinem until 4.03 were performed without SNR scaling
            if (self.training_parameters['snr_range'][0] != 1000 and avg_foreground_power != 0):
                # Normalize SNR to value
                # Add the noise to the foreground

                bgnoise = torch.mul(background_mul, k)[:,0]
                background_add = torch.add(bgnoise, sliced_foreground)

                # augmenting with "silence"
                if (torch.sum(background_mul_energy) == 0): # if the noise is null (== silence)
                    bgnoise = torch.mul(background_mul, 1)[:,0]
                    background_add = torch.add(bgnoise, sliced_foreground)
            else:
                # Add the noise * NF
                background_add = torch.add(background_mul[:,0], sliced_foreground)
                bgnoise = background_mul[:,0]

        else:
            background_add = sliced_foreground
            bgnoise = torch.from_numpy(np.zeros([self.preprocessing_parameters['desired_samples'], 1]))

        if (self.device.type == 'cuda'): 
            background_add = background_add.cuda().float() 

        self.bgnoise = bgnoise
        self.background_add = background_add


    def denoisify(self):

        # denoisify
        import noisereduce as nr
        import onnxruntime as ort
        import onnx


        # Model the noise: y_noise
        if (self.training_parameters['denoisify'] == 1):
            self.background_add = nr.reduce_noise(y=self.background_add.cpu(), y_noise = self.bgnoise.cpu(), sr=self.preprocessing_parameters['desired_samples'], \
            stationary = False)        
            self.background_add = torch.tensor(self.background_add).cuda()

        elif (self.training_parameters['denoisify'] == 2):
            self.background_add = nr.reduce_noise(y=self.background_add.cpu(), y_noise = self.bgnoise.cpu(), sr=self.preprocessing_parameters['desired_samples'], \
            stationary = True)
            self.background_add = torch.tensor(self.background_add).cuda()

        elif (self.training_parameters['denoisify'] == 3):

            data = self.background_add.cpu().numpy()
            
            gru = 1 # TODO: Parametrize
            win_len = 400
            win_inc = 100 
            fft_len = 512

            stft_frame_i = librosa.stft(
                data, win_length=win_len, 
                n_fft=fft_len, hop_length=win_inc,
                window='hann', center=False)
            fft_feat, num_win = stft_frame_i.shape

            stft_frame_i_T = np.transpose (stft_frame_i) # swap the axis to select the tmestamp
            stft_frame_o_T = np.empty_like(stft_frame_i_T)

            h_state_len = 256
            rnn_0_i_state = np.zeros(h_state_len)
            rnn_1_i_state = np.zeros(h_state_len)
            rnn_0_c_state = np.zeros(h_state_len)
            rnn_1_c_state = np.zeros(h_state_len)

            for mindex in range (num_win):
                stft_clip = stft_frame_i_T[mindex]
                stft_clip_mag = np.abs(stft_clip)

                if gru == 1:
                    data = [stft_clip_mag, rnn_0_i_state, rnn_1_i_state]
                else:
                    data = [stft_clip_mag, rnn_0_i_state, rnn_0_c_state, rnn_1_i_state, rnn_1_c_state]

                minput = np.array(stft_clip_mag.reshape((1, 257, 1))).astype(np.single)
                mstatein = np.array([rnn_0_i_state, rnn_1_i_state]).reshape((2, 1, 256)).astype(np.single)

                outputs = self.ort_session.run(
                    None,
                    {"input": minput, 
                     "state_in": mstatein },
                )                    
                
                mag_out = np.array(outputs[0][0])

                if gru == 1:
                    rnn_0_i_state = np.array(outputs[1][0]) # outputs[self.onnxmodel['GRU_74'].step_idx][0]
                    rnn_1_i_state = np.array(outputs[1][1]) # outputs[self.onnxmodel['GRU_136'].step_idx][0]
                else:
                    rnn_0_i_state = outputs[self.onnxmodel['LSTM_78'].step_idx][0]
                    rnn_0_c_state = outputs[self.onnxmodel['output_2'].step_idx][0]
                    rnn_1_i_state = outputs[self.onnxmodel['LSTM_144'].step_idx][0]
                    rnn_1_c_state = outputs[self.onnxmodel['output_3'].step_idx][0]

                stft_clip_mag_estimate = mag_out.squeeze()

                stft_clip = stft_clip * stft_clip_mag_estimate
                stft_frame_o_T[mindex] = stft_clip

            stft_frame_o = np.transpose (stft_frame_o_T)
            data = librosa.istft(stft_frame_o, hop_length=win_inc, 
                win_length=win_len, window='hann', center=False )

            curr_len = data.shape[0]

            data = np.array(data)
            data = np.pad(data, int((16000-curr_len)/2))

            self.background_add = torch.tensor(data).cuda()

        else:
            return

    # Preprocess samples to extract features
    def preprocess(self):

        if (self.environment_parameters['keywords_dataset'] == 'mswc'):
            melkwargs={ 'n_fft':2048, 'win_length':self.preprocessing_parameters['window_size_samples'], 'hop_length':self.preprocessing_parameters['window_stride_samples'],
             'f_min':20, 'f_max':4000, 'n_mels':self.preprocessing_parameters['n_mels']}
        else:
            melkwargs={ 'n_fft':1024, 'win_length':self.preprocessing_parameters['window_size_samples'], 'hop_length':self.preprocessing_parameters['window_stride_samples'],
                         'f_min':20, 'f_max':4000, 'n_mels':self.preprocessing_parameters['n_mels']}

        if (self.preprocessing_parameters['library'] == "pytorch"):
            import torchaudio

            if (self.device.type == 'cuda'):
                torch.set_default_tensor_type('torch.cuda.FloatTensor') 
            else:
                self.background_add = self.background_add.float()
            mfcc_transformation = torchaudio.transforms.MFCC(n_mfcc=self.preprocessing_parameters['feature_bin_count'], sample_rate=self.preprocessing_parameters['desired_samples'], melkwargs=melkwargs, log_mels=True, norm='ortho')
            data = mfcc_transformation(self.background_add)

            self.data_placeholder = torch.transpose(data[:,:self.preprocessing_parameters['spectrogram_length']], 0, 1)

            if (self.device.type == 'cuda'):
                torch.set_default_tensor_type('torch.FloatTensor')

        elif (self.preprocessing_parameters['library'] == "tensorflow"):
            import tensorflow as tf
            tf_data = tf.convert_to_tensor(self.background_add.cpu().numpy(), dtype=tf.float32)
            tf_stfts = tf.signal.stft(tf_data, frame_length=self.preprocessing_parameters['window_size_samples'], frame_step=self.preprocessing_parameters['window_stride_samples'], fft_length=1024)
            tf_spectrograms = tf.abs(tf_stfts)
            power = True
            if power:
                    tf_spectrograms = tf_spectrograms ** 2
            num_spectrogram_bins = tf_stfts.shape[-1]
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.preprocessing_parameters['n_mels'], num_spectrogram_bins, self.preprocessing_parameters['desired_samples'], 20, 4000)
            tf_spectrograms = tf.cast(tf_spectrograms, tf.float32)
            tf_mel_spectrograms = tf.tensordot(tf_spectrograms, linear_to_mel_weight_matrix, 1)
            tf_mel_spectrograms.set_shape(tf_spectrograms.shape[:-1].concatenate(
                                    linear_to_mel_weight_matrix.shape[-1:]))
            tf_log_mel = tf.math.log(tf_mel_spectrograms + 1e-6)
            tf_mfccs = tf.signal.mfccs_from_log_mel_spectrograms(tf_log_mel)[..., :self.preprocessing_parameters['feature_bin_count']]
            mfcc = torch.Tensor(tf_mfccs.numpy())
            self.data_placeholder = mfcc

        elif (self.preprocessing_parameters['library'] == "librosa"):
            import librosa
            S = librosa.feature.melspectrogram(y=self.background_add.numpy(), sr=self.preprocessing_parameters['desired_samples'], n_mels=self.preprocessing_parameters['n_mels'], fmin=20, fmax=4000, hop_length=320, win_length=640, n_fft=640, norm=None, htk=True, center=False, power=2)
            mfcc = librosa.feature.mfcc(S=np.log(S+1e-6), n_mfcc=self.preprocessing_parameters['feature_bin_count'], norm='ortho')[:self.preprocessing_parameters['feature_bin_count'], :]
            for col in range(0, mfcc.shape[1]):
                mfcc[0][col] = mfcc[0][col]*np.sqrt(2)
            mfcc = mfcc.T
            self.data_placeholder = mfcc

        else:
            raise ValueError("Preprocessing library not implemented.")


        # Shift data in [0, 255] interval to match Dory request for uint8 inputs
        self.data_placeholder = torch.clamp(self.data_placeholder + 128, 0, 255)

        # Removed for now, could prove useful when integrated with Quantlib
        # Normalize 
        # self.data_placeholder = (self.data_placeholder - torch.mean(self.data_placeholder)) / torch.std(self.data_placeholder)

        # Adding channel dimension
        self.data_placeholder = torch.reshape(self.data_placeholder, (-1, self.data_placeholder.size(dim=0), self.data_placeholder.size(dim=1)))

        # Prepare labels
        label_index = self.word_to_index[self.sample['label']]
        self.labels_placeholder = label_index



        # Prepare noise index
        complete_noise_list = {}

        if (self.task == -1 or self.training_parameters['noise_mode'] == 'noiseless'):
            self.noises_placeholder = -1
        else:
            # Return the index of the noise from the complete noise list
            # TODO: Parametrize the list to include other datasets
            # GSCv2
            complete_noise_list['gscv2'] = ['doing_the_dishes.wav', 'exercise_bike', 'white_noise', 'dude_miaowing', 'pink_noise', 'running_tap']
            # DEMAND
            complete_noise_list['demand'] = ['DKITCHEN', 'DLIVING', 'DWASHING', 'NFIELD', 'NPARK', \
                                                                    'NRIVER', 'OHALLWAY', 'OMEETING', 'OOFFICE', 'PCAFETER', \
                                                                    'PRESTO', 'PSTATION', 'SCAFE', 'SPSQUARE', 'STRAFFIC', \
                                                                    'TBUS', 'TCAR', 'TMETRO', 'SILENCE']
            # KINEM
            complete_noise_list['kinem'] = ['drone', 'kitchen', 'meeting', 'street', 'tester']

            if (self.mode == 'training'):
                self.noises_placeholder = complete_noise_list[self.offline_noise_train_dataset].index(self.offline_background_noise_train_name[self.data_augmentation_parameters['background_index']])
            elif (self.mode == 'validation'):
                self.noises_placeholder = complete_noise_list[self.offline_noise_test_dataset].index(self.offline_background_noise_test_name[self.data_augmentation_parameters['background_index']]) 
            elif (self.mode == 'odda'):
                self.noises_placeholder = complete_noise_list[self.online_noise_train_dataset].index(self.online_background_noise_train_name[self.data_augmentation_parameters['background_index']])
            elif (self.mode == 'odda_val'):
                self.noises_placeholder = complete_noise_list[self.online_noise_test_dataset].index(self.online_background_noise_test_name[self.data_augmentation_parameters['background_index']]) 
            elif (self.mode == 'testing'): # TODO: split in offline testing and online testing
                self.noises_placeholder = complete_noise_list[self.online_noise_test_dataset].index(self.online_background_noise_test_name[self.data_augmentation_parameters['background_index']]) 

        
    def __getitem__(self, idx):

        self.setup(idx)
        self.augment()
        # self.denoisify()
        self.preprocess()


        return self.data_placeholder, self.labels_placeholder, self.noises_placeholder, self.paths_placeholder

    


