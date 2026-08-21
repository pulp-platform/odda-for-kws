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


import torch
import os
import time
import math
import random
import sys
import argparse
import shutil
import csv
import glob

from datagenerator import DatasetCreator
from architectures.dscnn import DSCNN, DSCNNS, DSCNNM, DSCNNL
from train import Train
from quantize import nemo_quantize

from torchsummary import summary
from utils import parameter_generation
from copy import deepcopy
from pthflops import count_ops



def main():

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--feature_bin_count', type=int, default=None, help="Selected mels") 
    parser.add_argument('--time_shift_ms', type=int, default=None, help="Shifting for augmentation") 
    parser.add_argument('--sample_rate', type=int, default=None, help="Sample rate") 
    parser.add_argument('--clip_duration_ms', type=int, default=None, help="Input length") 
    parser.add_argument('--window_size_ms', type=int, default=None, help="Window len") 
    parser.add_argument('--window_stride_ms', type=int, default=None, help="Window hop")
    parser.add_argument('--n_mels', type=int, default=None, help="Number of mels")
    parser.add_argument('--noise_seed', type=int, default=None, help="Seed during training time")
         
    parser.add_argument('--device_id', type=str, default=None, help = 'Select GPU for use')
    parser.add_argument('--noise_dataset', type=str, default=None, help = "Noise dataset")
    parser.add_argument('--noise_dir_demand', type=str, default=None, help = "DEMAND noise path")
    parser.add_argument('--noise_dir_gscv2', type=str, default=None, help = "GSC noise path")
    parser.add_argument('--data_url', type=str, default=None, help = "Data URL")
    parser.add_argument('--keywords_dataset', type=str, default=None, help = "Keywords dataset")
    parser.add_argument('--data_dir_gscv2', type=str, default=None, help = "GSC noise path")
    parser.add_argument('--data_dir_mswc', type=str, default=None, help = "MSWC noise path")
    parser.add_argument('--data_dir_kinem', type=str, default=None, help = "KINEM noise path")
    parser.add_argument('--denoisify', type=int, default=None, help = 'Apply denoiser or not, and select the denoiser.')
    parser.add_argument('--reverb', type=str, default=None, help = 'Apply reverb')

    parser.add_argument('--noise_mode', type=str, default=None, help = 'noiseless, noiseaware, odda')
    parser.add_argument('--snr_range', nargs="*", type=int, default=None, help = 'SNR range, passed as list')
    parser.add_argument('--model', type=str, default=None, help = 'DSCNN')
    parser.add_argument('--channels', type=int, default=None, help = '64/172/276')
    parser.add_argument('--blocks', type=int, default=None, help = '4/4/5')
    parser.add_argument('--use_pretrained', type=int, default=None, help = 'Use pretrained model')
    parser.add_argument('--pretrained_directory', type=str, default=None, help = 'Location of pretrained model')
    parser.add_argument('--epochs', type=int, default=None, help = 'Number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, help = 'Batch size')
    parser.add_argument('--bnstat_update', type=bool, default=None, help = 'Update running stats of BatchNorm')
    parser.add_argument('--loss', type=str, default=None, help = 'Loss type')
    parser.add_argument('--initial_lr', type=float, default=None, help = 'Initial learning rate')
    parser.add_argument('--optimizer', type=str, default=None, help = 'Optimizer')
    parser.add_argument('--momentum', type=float, default=None, help = 'Momentum')
    parser.add_argument('--frozen_layers', nargs="*", type=str, default=None, help = 'List of frozen layers, for instance: "conv1", "bn1"...')
    parser.add_argument('--silence_percentage', type=int, default=None, help = 'Percentage of silence in the dataset')
    parser.add_argument('--unknown_percentage', type=int, default=None, help = 'Percentage of unknown in the dataset')
    parser.add_argument('--validation_percentage', type=int, default=None, help = 'Percentage of validation data in the dataset')
    parser.add_argument('--testing_percentage', type=int, default=None, help = 'Percentage of test data in the dataset')
    parser.add_argument('--background_frequency', type=int, default=None, help = 'Frequency of adding background noise. Default is 1 in ODDA.')
    parser.add_argument('--background_volume', type=int, default=None, help = 'Background noise volume. Argument is ignored if snr_range is set')
    parser.add_argument('--task', type=str, default=None, help='Selected task')
    parser.add_argument('--wanted_words_gscv2_12w', nargs="*", type=str, default=None, help='List of GSC12 words')
    parser.add_argument('--wanted_words_gscv2_35w', nargs="*", type=str, default=None, help='List of GSC35 words')
    parser.add_argument('--wanted_frequency_mswc', type=str, default=None, help='Threshold in samples/class to select MSWC words')

    parser.add_argument('--train', type=int, default=None, help='Perform (pre)training model')
    parser.add_argument('--metatrain', type=int, default=None, help='Perform metatraining for the model')
    parser.add_argument('--select', type=int, default=None, help='Perform data selection for ODDA')
    parser.add_argument('--odda', type=int, default=None, help='Perform ODDA')
    parser.add_argument('--quantize', type=int, default=None, help='Quantize model. Not implemented.')
    parser.add_argument('--evaluate', type=int, default=None, help='Evaluate pretrained model.')

    parser.add_argument('--selection_method', type=str, default = None, help = 'Data selection method')
    parser.add_argument('--selection_interval_upper', type=float, default = None, help= 'Data selection interval upper bound')
    parser.add_argument('--selection_interval_lower', type=float, default = None, help= 'Data selection interval lower bound')

    parser.add_argument('--noise_train', nargs="*", type=str, default=None, help = 'List of noises on which we train the net: "DKITCHEN", "DLIVING"...')
    parser.add_argument('--noise_test', nargs="*", type=str, default=None, help = 'List of noises on which we test the net: "DKITCHEN", "DLIVING"....')
    parser.add_argument('--target_noise', type=str, default = None, help = 'ODDA on-site noise, select from the list above')
    parser.add_argument('--distance', type=str, default = None, help = 'Select microphone-to-subject distance. N(ear)/F(ar).')
    parser.add_argument('--microphone', type=str, default = None, help = 'Select microphone quality. L(ow-power)/P(hone)/R(hode).')

    parser.add_argument('--base_path', type=str, default = None, help='Path to current directory')
    parser.add_argument('--model_path', type=str, default = None, help='Path where the model will be saved')

    parser.add_argument('--config_file', type=str, default='config.json', help = 'Configuration file')
    

    args = vars(parser.parse_args())

    # Parameter generation
    environment_parameters, preprocessing_parameters, training_parameters, experimental_parameters = parameter_generation(args) 


    # Device setup
    os.environ["CUDA_VISIBLE_DEVICES"] = environment_parameters['device_id']
    if torch.cuda.is_available() and environment_parameters['device'] == 'gpu':
        device = torch.device('cuda')        
    else:
        device = torch.device('cpu')
    print (torch.version.__version__)
    print (device)

    # Dataset generation
    audio_processor = DatasetCreator(environment_parameters, training_parameters, preprocessing_parameters, experimental_parameters)

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    odda_size = audio_processor.get_size('odda')
    print("Dataset split (Train/valid/test/ODDA): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size) + "/" + str(odda_size))

    # Model generation and analysis
    n_classes = len(audio_processor.words_list)
    print (audio_processor.words_list)
    print (n_classes)
    
    # UP TO DATE
    model = getattr(sys.modules["architectures.dscnn"], training_parameters['model'])(n_channels = training_parameters['channels'], 
        n_blocks = training_parameters['blocks'], n_classes = 12, use_bias = False, stem = 'sym', padding='asym', device = device)

    # NOT SO OLD
    # model = getattr(sys.modules["architectures.dscnn"], training_parameters['model'])(n_channels = training_parameters['channels'], 
    #     n_blocks = training_parameters['blocks'], n_classes = 12, use_bias = True, stem = 'asym', padding='sym', device = device)
        
    # OUTDATED
    # model = getattr(sys.modules["architectures.dscnn"], training_parameters['model'])(use_bias = True)
    
    model.to(device)

    # Freeze layers
    # Extend frozen layers list - weights and biases
    frozen_layers = []
    for elem in training_parameters['frozen_layers']:
        # Outdated
        # if "." in elem:
        #     frozen_layers.append(elem)
        # else:
        #     frozen_layers.append(elem+".weight")
        #     frozen_layers.append(elem+".bias")
        # IOTJ25
        frozen_layers.append(elem+".weight")
        frozen_layers.append(elem+".bias")
       
    # Freeze layers
    for layer_name, layer_param in model.named_parameters():
        print (layer_name)
        if layer_name in frozen_layers:
            print ("Freezing")
            print (layer_name)
            layer_param.requires_grad = False

    print (model)

    summary(model,(1,49,preprocessing_parameters['feature_bin_count']), device=device.type)
    dummy_input = torch.rand(1, 1,49,preprocessing_parameters['feature_bin_count']).to(device)
    count_ops(model, dummy_input)

    train_log_path = experimental_parameters['model_path']+'nakws/'+ training_parameters['model'] + "_" + str(experimental_parameters['target_noise']) + \
                    "_" + experimental_parameters['date'] + "/"

    # Training initialization
    training_environment = Train(audio_processor, training_parameters, model, device, train_log_path)

    if experimental_parameters['train']:

        print (preprocessing_parameters)
        print (training_parameters)
        print (environment_parameters)
        print (experimental_parameters)
        
        # Train
        start=time.clock_gettime(0)
        training_environment.train(model) 
        print('Finished Training on GPU in {:.2f} seconds'.format(time.clock_gettime(0)-start))

        with open(train_log_path+'log.csv','a') as csvfile:    
            writer = csv.writer(csvfile, delimiter=',')
            # Gives the header name row into csv
            for key, value in environment_parameters.items():
                writer.writerow([key, value]) 
            for key, value in preprocessing_parameters.items():
                writer.writerow([key, value]) 
            for key, value in training_parameters.items():
                writer.writerow([key, value]) 
            for key, value in experimental_parameters.items():
                writer.writerow([key, value]) 

        shutil.copyfile(args['config_file'], train_log_path+args['config_file'])

        # Accuracy on the validation set. 
        print ("Validation acc")
        acc = training_environment.validate(model, mode='validation', statistics=False)
        # Accuracy on the testing set. 
        # print ("Testing acc")
        # acc = training_environment.validate(model, mode='testing', statistics=False)

    if (training_parameters['use_pretrained']):
        # Per-epoch analysis
        model.load_state_dict(torch.load(training_parameters['pretrained_directory']+'/model.pth', map_location=device))


    if (experimental_parameters['evaluate']):

        odda_training_parameters = training_parameters
        odda_environment_parameters = environment_parameters
        odda_environment_parameters['noise_train'] = experimental_parameters['target_noise']
        odda_environment_parameters['noise_test'] = experimental_parameters['target_noise']

        print ("----------------------")
        print (odda_training_parameters)
        print (odda_environment_parameters)
  
        # Generate new dataset considering ODDA constraints
        odda_audio_processor = DatasetCreator(odda_environment_parameters, odda_training_parameters, preprocessing_parameters, experimental_parameters)
        # Create training and evaluation environment
        odda_environment = Train(odda_audio_processor, odda_training_parameters, model, device, train_log_path) 

        # Accuracy on the validation set. 
        print ("Validation acc")
        acc = odda_environment.validate(model, mode='validation', statistics=False)

        # Accuracy on the ODDA validation set. 
        print ("Validation acc on TARGET noise")
        acc = odda_environment.validate(model, mode='odda_val', statistics=False)


    odda_size = audio_processor.get_size('odda')
    print ("ODDA set size: " + str(odda_size))

    train_size = audio_processor.get_size('training')
    valid_size = audio_processor.get_size('validation')
    test_size = audio_processor.get_size('testing')
    odda_size = audio_processor.get_size('odda')
    print("Dataset split (Train/valid/test/ODDA): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size) + "/" + str(odda_size))

    if (training_parameters['frozen_layers']): # 0 instead of -1 for inversion
        if (training_parameters['inverted']):
            odda_log_path = experimental_parameters['model_path']+'odda/'+ training_parameters['model'] + "_" + experimental_parameters['selection_method']+ "_" + \
                    str(experimental_parameters['target_noise']) +"_" + str(training_parameters['frozen_layers'][0]) + "_"+ \
                    str(experimental_parameters['selection_interval_lower']) + \
                    "_" + str(experimental_parameters['selection_interval_upper']) + "_" + str(training_parameters['snr_range']).replace(" ", "").replace(",","_") + "_" + experimental_parameters['date'] + "/"
        else:
            odda_log_path = experimental_parameters['model_path']+'odda/'+ training_parameters['model'] + "_" + experimental_parameters['selection_method']+ "_" + \
                    str(experimental_parameters['target_noise']) +"_" + str(training_parameters['frozen_layers'][-1]) + "_"+ \
                    str(experimental_parameters['selection_interval_lower']) + \
                    "_" + str(experimental_parameters['selection_interval_upper']) + "_" + str(training_parameters['snr_range']).replace(" ", "").replace(",","_") + "_" + experimental_parameters['date'] + "/"

    else:
        odda_log_path = experimental_parameters['model_path']+'odda/'+ training_parameters['model'] + "_" + experimental_parameters['selection_method']+ "_" + \
                        str(experimental_parameters['target_noise']) + "_" + \
                        str(experimental_parameters['selection_interval_lower']) + \
                        "_" + str(experimental_parameters['selection_interval_upper']) + "_" + str(training_parameters['snr_range']).replace(" ", "").replace(",","_") + "_" + experimental_parameters['date'] + "/"

    # Perform ODDA
    if experimental_parameters['odda']:

        # Select ODDA noise
        odda_training_parameters = training_parameters
        odda_training_parameters['initial_lr'] = training_parameters['initial_lr']
        
        odda_environment_parameters = environment_parameters
        odda_environment_parameters['noise_train'] = experimental_parameters['target_noise']
        odda_environment_parameters['noise_test'] = experimental_parameters['target_noise']

        print ("----------------------")
        print (odda_training_parameters)
        print (odda_environment_parameters)
  
        # Generate new dataset considering ODDA constraints
        odda_audio_processor = DatasetCreator(odda_environment_parameters, odda_training_parameters, preprocessing_parameters, experimental_parameters)

        # Hacky 'odda' dataset transfer from selection step to adaptation step
        # TODO: Integrate
        odda_audio_processor.data_set['odda'] = deepcopy(audio_processor.data_set['odda'])

        # Generate training environment using new parameters
        odda_environment = Train(odda_audio_processor, odda_training_parameters, model, device, odda_log_path) 

        # Perform pre-odda evaluation
        # Accuracy on the validation set. 
        print ("Validation acc pre-ODDA")
        acc = odda_environment.validate(model, mode='validation', statistics=False)
        # Accuracy on the ODDA validation set. 
        print ("Validation acc pre-ODDA on TARGET noise")
        acc = odda_environment.validate(model, mode='odda_val', statistics=False)

        odda_environment.adapt(model)

        with open(odda_log_path+'log.csv','a') as csvfile:    
            writer = csv.writer(csvfile, delimiter=',')
            # Gives the header name row into csv
            for key, value in odda_environment_parameters.items():
                writer.writerow([key, value]) 
            for key, value in preprocessing_parameters.items():
                writer.writerow([key, value]) 
            for key, value in odda_training_parameters.items():
                writer.writerow([key, value]) 
            for key, value in experimental_parameters.items():
                writer.writerow([key, value]) 

        shutil.copyfile(args['config_file'], odda_log_path+args['config_file'])

        # Final accuracy on ODDA
        print ("Validation acc")
        acc = odda_environment.validate(model, mode='validation', statistics=False)
        # Accuracy on the ODDA validation set. 
        print ("Validation acc on TARGET noise")
        acc = odda_environment.validate(model, mode='odda_val', statistics=False)

    # # Accuracy on the training set. 
    # print ("Training acc")
    # acc = training_environment.validate(model, mode='training', batch_size=-1, statistics=False)
    # Accuracy on the validation set. 
    print ("Validation acc")
    acc = training_environment.validate(model, mode='validation', statistics=False)
    # TODO: Fix OFFLINE vs ONLINE testing
    # Accuracy on the testing set. 
    # print ("Testing acc")
    # acc = training_environment.validate(model, mode='testing', statistics=False)
    # Accuracy on the ODDA validation set. 
    print ("Validation acc on TARGET noise")
    acc = training_environment.validate(model, mode='odda_val', statistics=False)

    if experimental_parameters['quantize']:
        nemo_quantize(device, model)

    print ("Experiment complete.")

    # Move all experiment-related files to the directory

if __name__ == "__main__":
    main()
