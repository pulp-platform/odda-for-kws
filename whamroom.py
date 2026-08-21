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


import numpy as np
import pyroomacoustics as pra

from pyroomacoustics.parameters import constants
from scipy.signal import resample_poly

class ReverbRoom(pra.room.ShoeBox):

    def __init__(self, p, mics, s1, T60, fs=16000,
                 t0=0., sigma2_awgn=None):

        self.T60 = T60
        self.max_rir_len = np.ceil(T60*fs).astype(int)

        volume = p[0]*p[1]*p[2]
        surface_area = 2*(p[0]*p[1] + p[0]*p[2] + p[1]*p[2])
        absorption = 24 * volume * np.log(10.0) / (constants.get('c') * surface_area * T60)

        # minimum max order to guarantee complete filter of length T60
        max_order = np.ceil(T60 * constants.get('c') / min(p)).astype(int)

        super().__init__(p, fs=fs, t0=t0, absorption=absorption,
                         max_order=max_order, sigma2_awgn=sigma2_awgn,
                         sources=None, mics=None)

        self.add_source(s1)

        self.add_microphone_array(pra.MicrophoneArray(np.c_[mics], fs))


    def add_audio(self, s1):
        self.sources[0].add_signal(s1)
        

    def generate_rirs(self):

        self.compute_rir()
        self.rir_reverberant = self.rir


    def generate_audio(self, anechoic=False, fs=16000):

        if not self.rir:
            self.generate_rirs()
        if anechoic:
            self.rir = self.rir_anechoic
        else:
            self.rir = self.rir_reverberant
        audio_array = self.simulate(return_premix=True, recompute_rir=False)

        if type(fs) is not list:
            fs_array = [fs]
        else:
            fs_array = fs
        audio_out = []
        for elem in fs_array:
            if type(elem) is str:
                elem = int(elem.replace('k','000'))
            if elem != self.fs:
                assert(self.fs % elem == 0)
                audio_out.append(resample_poly(audio_array, elem, self.fs, axis=2))
            else:
                audio_out.append(audio_array)
        if type(fs) is not list:
            return audio_out[0] # array of shape (n_sources, n_mics, n_samples)
        else:
            return audio_out


class AnechoicRoom(pra.room.AnechoicRoom):

    def __init__(self, p, mics, s1, fs=16000,
                 t0=0., sigma2_awgn=None):

        super().__init__(p, fs=fs, t0=t0)

        self.add_source(s1)

        self.add_microphone_array(pra.MicrophoneArray(np.c_[mics], fs))


    def add_audio(self, s1):
        self.sources[0].add_signal(s1)

    def generate_rirs(self):

        self.compute_rir()
        self.rir_anechoic = self.rir

    def generate_audio(self, anechoic=True, fs=16000):

        if not self.rir:
            self.generate_rirs()
        self.rir = self.rir_anechoic
        audio_array = self.simulate(return_premix=True, recompute_rir=False)

        if type(fs) is not list:
            fs_array = [fs]
        else:
            fs_array = fs
        audio_out = []
        for elem in fs_array:
            if type(elem) is str:
                elem = int(elem.replace('k','000'))
            if elem != self.fs:
                assert(self.fs % elem == 0)
                audio_out.append(resample_poly(audio_array, elem, self.fs, axis=2))
            else:
                audio_out.append(audio_array)
        if type(fs) is not list:
            return audio_out[0] # array of shape (n_sources, n_mics, n_samples)
        else:
            return audio_out
