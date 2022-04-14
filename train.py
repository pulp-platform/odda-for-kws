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


import dataset
import torch

from utils import confusion_matrix, npy_to_txt, per_noise_accuracy

import torch.nn.functional as F


class Train():

    def __init__(self, audio_processor, training_parameters, model, device):
        self.audio_processor = audio_processor
        self.training_parameters = training_parameters
        self.model = model
        self.device = device

        # Training hyperparameters
        self.criterion = torch.nn.CrossEntropyLoss()
        intitial_lr = 0.001
        self.optimizer = torch.optim.Adam(model.parameters(), lr = intitial_lr)
        lambda_lr = lambda epoch: 1 if epoch<15 else 1/5 if epoch < 25 else 1/10 if epoch<35 else 1/20
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda_lr)


    def validate(self, model = None, mode='validation', batch_size = 0, statistics = False, integer = False, save = False, task = None):
        # Validate model

        training_parameters = self.training_parameters
        if (batch_size != 0):
            training_parameters['batch_size'] = batch_size   


        data = dataset.AudioGenerator(mode, self.audio_processor, training_parameters, task = task)
        model.eval()  

        correct = 0
        total = 0

        with torch.no_grad():

            if batch_size == -1:
                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).long().to(self.device)
                model = model.to(self.device)  

                if (integer):
                    model = model.cpu()
                    inputs = inputs * 255./255 
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                if (save):
                    model = model.cpu()
                    inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                    outputs = F.softmax(model(inputs, save), dim=1)
                    outputs = outputs.to(self.device)
                    npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                else:
                    outputs = F.softmax(model(inputs), dim=1)
                    outputs = outputs.to(self.device)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                if statistics == True:
                    # conf_matrix(labels, predicted, self.training_parameters)
                    per_noise_accuracy(labels, predicted, noises)

            else:

                for minibatch in range(len(data)):
                    inputs, labels, noises = data[0]
                    inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                    labels = torch.Tensor(labels).long().to(self.device)
                    model = model.to(self.device)  

                    if (integer):
                        model = model.cpu()
                        inputs = inputs * 255./255 
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()           

                    if (save):
                        model = model.cpu()
                        inputs = inputs.type(torch.uint8).type(torch.float).cpu()
                        outputs = F.softmax(model(inputs, save), dim=1)
                        outputs = outputs.to(self.device)
                        npy_to_txt(-1, inputs.int().cpu().detach().numpy())
                    else:
                        outputs = F.softmax(model(inputs), dim=1)
                        outputs = outputs.to(self.device)

                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    if minibatch % 20 == 0: 
                        print('[%3d / %3d] accuracy: %.3f' % (minibatch + 1, len(data),  100 * correct / total))
                        running_loss = 0.0

                    if statistics == True:
                        # conf_matrix(labels, predicted, self.training_parameters)
                        per_noise_accuracy(labels, predicted, noises)

        print('Accuracy of the network on the %s set: %.2f %%' % (mode, 100 * correct / total))
        return(100 * correct / total)


    def train(self, model):
        # Train model

        best_acc = 0
        for epoch in range(0, self.training_parameters['epochs']):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task = None)
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            for minibatch in range(len(data)):

                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print information every 20 minibatches
                if minibatch % 20 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation')

        PATH = './model.pth'
        torch.save(model.state_dict(), PATH)

    def adapt (self, model, noise):

        self.training_parameters['noise_test'] = noise
        self.training_parameters['noise_train'] = noise

         # Train model
        best_acc = 0
        best_ep  = 0
        for epoch in range(0, self.training_parameters['epochs']):

            print("Epoch: " + str(epoch+1) +"/" + str(self.training_parameters['epochs']))

            # task = -1      - Silent training
            # task = None    - Mixed training
            # task = epoch-1 - NaiveCL
            # task = n       - Fixed-noise training

            data = dataset.AudioGenerator('training', self.audio_processor, self.training_parameters, task = None)
            model.train()
            self.scheduler.step()

            running_loss = 0.0
            total = 0
            correct = 0   

            for minibatch in range(len(data)):

                inputs, labels, noises = data[0]
                inputs = torch.Tensor(inputs[:,None,:,:]).to(self.device)
                labels = torch.Tensor(labels).to(self.device).long()

                # Zero out the parameter gradients after each mini-batch
                self.optimizer.zero_grad()

                # Train, compute loss, update optimizer
                model = model.to(self.device)
                outputs = F.softmax(model(inputs), dim=1)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # Compute training statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Print information every 20 minibatches
                if minibatch % 20 == 0: 
                    print('[%3d / %3d] loss: %.3f  accuracy: %.3f' % (minibatch + 1, len(data), running_loss / 10, 100 * correct / total))
                    running_loss = 0.0

            tmp_acc = self.validate(model, 'validation', task = 0)

            # Save best performing network
            if (tmp_acc > best_acc):
                best_acc = tmp_acc
                best_ep  = epoch
                BEST_PATH = './model_ep' + str(best_ep) + '_acc' + str(best_acc) + '.pth'
                torch.save(model.state_dict(), BEST_PATH)

            patience = 10
            if (epoch >= best_ep + patience):
              break


        PATH = './model.pth'
        torch.save(model.state_dict(), PATH)

        
