# Copyright (C) 2022 ETH Zurich
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
# Author: Cristian Cioflan, ETH (cioflanc@iis.ee.ethz.ch)


import torch
import dataset
import os
import time
import math
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from torchsummary import summary
from model import DSCNNS, DSCNNM, DSCNNL
from utils import remove_txt, parameter_generation
from copy import deepcopy
from pthflops import count_ops
from train import Train

# Device setup
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print (torch.version.__version__)
print(device)

# Parameter generation
training_parameters, data_processing_parameters = parameter_generation()  # To be parametrized

# Dataset generation
audio_processor = dataset.AudioProcessor(training_parameters, data_processing_parameters)

train_size = audio_processor.get_size('training')
valid_size = audio_processor.get_size('validation')
test_size = audio_processor.get_size('testing')
print("Dataset split (Train/valid/test): "+ str(train_size) +"/"+str(valid_size) + "/" + str(test_size))

# Model generation and analysis
model = DSCNNS(use_bias = True)
model.to(device)
summary(model,(1,49,data_processing_parameters['feature_bin_count']))
dummy_input = torch.rand(1, 1,49,data_processing_parameters['feature_bin_count']).to(device)
count_ops(model, dummy_input)

# Training initialization
training_environment = Train(audio_processor, training_parameters, model, device)

# Removing stored inputs and activations
remove_txt()

train = False
if train:
  # Train
  start=time.clock_gettime(0)
  training_environment.train(model)
  print('Finished Training on GPU in {:.2f} seconds'.format(time.clock_gettime(0)-start))

# Perform ODDA
odda = True
if odda:
  # Load pretrained model
  model.load_state_dict(torch.load('./pretrain2_ordered_v3_fixFilfixUttr.pth', map_location=torch.device('cuda')))
  environmental_noise = 'TMETRO'
  training_environment.adapt(model, environmental_noise)

# # Accuracy on the training set. 
# print ("Training acc")
# acc = training_environment.validate(model, mode='training', batch_size=-1, statistics=False)
# Accuracy on the validation set. 
# print ("Validation acc")
# acc = training_environment.validate(model, mode='validation', batch_size=-1, statistics=False)
# Accuracy on the testing set. 
print ("Testing acc")
acc = training_environment.validate(model, mode='testing', batch_size=-1, statistics=False)

