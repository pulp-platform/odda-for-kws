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

import nemo
import torch

from utils import remove_txt
from copy import deepcopy


# TODO: parametrize
def nemo_quantize(device, model, model_path = 'model', precision = 8):

    if quantize:

        # Load pretrained model
        if (model_path):
        	model.load_state_dict(torch.load(model_path+'.pth', map_location=device))

        # Initiating quantization process: making the model quantization aware
        quantized_model = nemo.transform.quantize_pact(deepcopy(model), dummy_input=torch.randn((1,1,49,10)).to(device))

        precision_dict = {
                            "conv1": {
                                "W_bits": precision-1
                            },
                            "relu1": {
                                "x_bits": precision
                            },
                            "conv2": {
                                "W_bits": precision-1
                            },
                            "relu2": {
                                "x_bits": precision
                            },
                            "conv3": {
                                "W_bits": precision-1
                            },
                            "relu3": {
                                "x_bits": precision
                            },
                            "conv4": {
                                "W_bits": precision-1
                            },
                            "relu4": {
                                "x_bits": precision
                            },
                            "conv5": {
                                "W_bits": precision-1
                            },
                            "relu5": {
                                "x_bits": precision
                            },
                            "conv6": {
                                "W_bits": precision-1
                            },
                            "relu6": {
                                "x_bits": precision
                            },
                            "conv7": {
                                "W_bits": precision-1
                            },
                            "relu7": {
                                "x_bits": precision
                            },
                            "conv8": {
                                "W_bits": precision-1
                            },
                            "relu8": {
                                "x_bits": precision
                            },
                            "conv9": {
                                "W_bits": precision-1
                            },
                            "relu9": {
                                "x_bits": precision
                            },
                            "fc1": {
                                "W_bits": precision-1
                            }

                }
        quantized_model.change_precision(bits=1, min_prec_dict=precision_dict, scale_weights=True, scale_activations=True)

        # Calibrating model's scaling by collecting largest activations
        with quantized_model.statistics_act():
                training_environment.validate(model=quantized_model, mode='validation', batch_size=128)
        quantized_model.reset_alpha_act()

        # Remove biases after FQ stage
        quantized_model.remove_bias()

        print("\nFakeQuantized @ 8b accuracy (calibrated):")
        acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1)

        quantized_model.qd_stage(eps_in=255./255)    # The activations are already in 0-255

        print("\nQuantizedDeployable @ mixed-precision accuracy:")
        acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1)

        quantized_model.id_stage()

        print("\nIntegerDeployable @ mixed-precision accuracy:")
        acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=-1, integer=True)

        # Saving the model
        nemo.utils.export_onnx(model_path + '.onnx', quantized_model, quantized_model, (1, 49, 10))
        # Saving the activations for comparison within Dory
        acc = training_environment.validate(model=quantized_model, mode='testing', batch_size=1, integer=True, save=True)

# TODO: Quantlib quantization