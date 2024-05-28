# Copyright (C) 2021-2023 ETH Zurich
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
# Adapted from: Maciejewski, Matthew and Wichern, Gordon and Le Roux, Jonathan, "WHAMR!"
# Author: Cristian Cioflan, ETH Zurich (cioflanc@iis.ee.ethz.ch)

import time

import numpy as np
import soundfile as sf
import pyroomacoustics as pra

from numpy.random import uniform
from wham_room import ReverbRoom, AnechoicRoom


def draw_params(reverb_level):

    # WHAMR!
    # reverb_room_dim = np.array([uniform(5, 10),
    #                      uniform(5, 10),
    #                      uniform(3, 4)])

    # center = np.array([reverb_room_dim[0]/2 + uniform(-0.2, 0.2),
    #                    reverb_room_dim[1]/2 + uniform(-0.2, 0.2),
    #                    uniform(0.9, 1.8)])

    reverb_room_dim = np.array([uniform(4, 5),
                         uniform(4, 5),
                         uniform(2, 2.5)])

    center = np.array([reverb_room_dim[0]/2 + uniform(-0.2, 0.2),
                       reverb_room_dim[1]/2 + uniform(-0.2, 0.2),
                       reverb_room_dim[2]/2 + uniform(-0.1, 0.1)])

    mics = np.array([center])

    # WHAMR!
    # s1_dist = uniform(0.66, 2)
    # s1_theta = uniform(0, 2*np.pi)
    # s1_height = uniform(0.9, 1.8)

    s1_dist = uniform(0.66, 0.99)
    s1_theta = uniform(0, 2*np.pi)
    s1_height = uniform(0.9, 1.8)

    s1_offset = np.array([np.cos(s1_theta) * s1_dist,
                          np.sin(s1_theta) * s1_dist,
                          s1_height - center[2]])
    s1 = center + s1_offset

    if reverb_level == "extreme":
        T60 = uniform(0.8, 1.0)
    if reverb_level == "high":
        T60 = uniform(0.6, 0.8)
    elif reverb_level == "medium":
        T60 = uniform(0.2, 0.6)
    elif reverb_level == "low":
        T60 = uniform(0.1, 0.3)

    return [reverb_room_dim, mics, s1, T60]


# Generate reverberous and anechoic rooms, compute RIRs, and augment the input signal
def gen_room_and_signal(signal, fs):

    start_time = time.time()
    reverb_room_params = draw_params(reverb_level='high')
    reverb_room_dim = reverb_room_params[0]
    mics = reverb_room_params[1]
    s1 = reverb_room_params[2]
    T60 = reverb_room_params[3]

    param_dict = { 'reverb_room_x' : reverb_room_dim[0],
                   'reverb_room_y' : reverb_room_dim[1],
                   'reverb_room_z' : reverb_room_dim[2],
                   'micL_x' : mics[0][0],
                   'micL_y' : mics[0][1],
                   'mic_z' : mics[0][2],
                   's1_x' : s1[0],
                   's1_y' : s1[1],
                   's1_z' : s1[2],
                   'T60' : T60 }

    print ("Param generation: " + str(time.time() - start_time))

    reverb_room = ReverbRoom([param_dict['reverb_room_x'], param_dict['reverb_room_y'], param_dict['reverb_room_z']],
                            [param_dict['micL_x'], param_dict['micL_y'], param_dict['mic_z']],
                            [param_dict['s1_x'], param_dict['s1_y'], param_dict['s1_z']],
                            param_dict['T60'])
    print ("ReverbRoom generation: " + str(time.time() - start_time))
    reverb_room.generate_rirs()
    print ("RIRs generation: " + str(time.time() - start_time))
    reverb_room.add_audio(signal)
    reverberant = reverb_room.generate_audio(fs=fs)
    print ("reverberant generation: " + str(time.time() - start_time))

    anechoic_room = AnechoicRoom(3,
                            [param_dict['micL_x'], param_dict['micL_y'], param_dict['mic_z']],
                            [param_dict['s1_x'], param_dict['s1_y'], param_dict['s1_z']])

    print ("AnechoicRoom generation: " + str(time.time() - start_time))
    anechoic_room.generate_rirs()
    anechoic_room.add_audio(signal)
    anechoic = anechoic_room.generate_audio(anechoic=True, fs=fs)
    print ("anechoic generation: " + str(time.time() - start_time))
    
    # Make relative source energy of anechoic sources same with original in mono (left channel) case
    # s1_spatial_scaling = np.sqrt(np.sum(signal ** 2) / np.sum(anechoic[0, 0, :] ** 2))
    signal_energy = np.sum(signal ** 2)
    anechoic_energy = np.sum(anechoic ** 2)
    if(anechoic_energy):
        s1_spatial_scaling = np.sqrt(np.sum(signal ** 2) / np.sum(anechoic ** 2))
    else:
        s1_spatial_scaling = 1

    s1_anechoic = anechoic[0, 0, :fs].T * s1_spatial_scaling
    s1_reverb = reverberant[0, 0, :fs].T * s1_spatial_scaling

    print ("signal generation: " + str(time.time() - start_time))

    return s1_reverb


def gen_room_params():

    room_params = draw_params(reverb_level='high')
    room_dim = room_params[0]
    mics = room_params[1]
    s1 = room_params[2]
    T60 = room_params[3]

    param_dict = { 'reverb_room_x' : room_dim[0],
                   'reverb_room_y' : room_dim[1],
                   'reverb_room_z' : room_dim[2],
                   'micL_x' : mics[0][0],
                   'micL_y' : mics[0][1],
                   'mic_z' : mics[0][2],
                   's1_x' : s1[0],
                   's1_y' : s1[1],
                   's1_z' : s1[2],
                   'T60' : T60 }

    return param_dict


# Generate randomly-characterized reverb room
def gen_reverb_room(param_dict):

    reverb_room = ReverbRoom([param_dict['reverb_room_x'], param_dict['reverb_room_y'], param_dict['reverb_room_z']],
                            [param_dict['micL_x'], param_dict['micL_y'], param_dict['mic_z']],
                            [param_dict['s1_x'], param_dict['s1_y'], param_dict['s1_z']],
                            param_dict['T60'])
    reverb_room.generate_rirs()

    return reverb_room


# Generate anechoic room
def gen_anechoic_room(param_dict):

    anechoic_room = AnechoicRoom(3,
                            [param_dict['micL_x'], param_dict['micL_y'], param_dict['mic_z']],
                            [param_dict['s1_x'], param_dict['s1_y'], param_dict['s1_z']])

    anechoic_room.generate_rirs()
    return anechoic_room

# Augment input signal in reverb room
def gen_signal(reverb_room, anechoic_room, signal, fs):

    reverb_room.add_audio(signal)
    reverberant = reverb_room.generate_audio(fs=fs)

    anechoic_room.add_audio(signal)
    anechoic = anechoic_room.generate_audio(anechoic=True, fs=fs)

    signal_energy = np.sum(signal ** 2)
    anechoic_energy = np.sum(anechoic ** 2)
    if(anechoic_energy):
        s1_spatial_scaling = np.sqrt(np.sum(signal ** 2) / np.sum(anechoic ** 2))
    else:
        s1_spatial_scaling = 1

    s1_anechoic = anechoic[0, 0, :fs].T * s1_spatial_scaling
    s1_reverb = reverberant[0, 0, :fs].T * s1_spatial_scaling

    return s1_reverb


    



