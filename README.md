# On-device Domain Adaptation for Noise-Robust Keyword Spotting

## Introduction

On-device Domain Adaptation (ODDA) for Noise-Robust Keyword Spotting is a methodology aimed at increasing the robustness to unseen noises for a keyword spotting system. The objective of keyword spotting (KWS) is to detect a set of predefined keywords within a stream of user utterances. The difficulty of the task increases in real environments with significant noise. To improve the performance of a KWS system in noise conditions unseen during training, we propose a methodology for tailoring a model to on-site noises through ODDA.

### Citing

If you use our methodology in an academic context, please cite the following publication:

Publications:
* *Efficient On-Device Domain Learning for Keyword Spotting on Ultra-Low-Power Platforms* [IEEE IOTJ 2026](https://ieeexplore.ieee.org/abstract/document/11352794)
* *On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems* [IEEE AICAS 2024](https://ieeexplore.ieee.org/abstract/document/10595987)
* *Towards On-device Domain Adaptation for Noise-Robust Keyword Spotting* [IEEE AICAS 2022](https://ieeexplore.ieee.org/document/9869990)

```
@ARTICLE{cioflan2026efficientondevice,
  author={Cioflan, Cristian and Cavigelli, Lukas and Rusci, Manuele and de Prado, Miguel and Benini, Luca},
  journal={IEEE Internet of Things Journal}, 
  title={Efficient On-Device Domain Learning for Keyword Spotting on Ultra-Low-Power Platforms}, 
  year={2026},
  volume={13},
  number={6},
  pages={10301-10316},
  keywords={Noise;Accuracy;Training;Noise robustness;Adaptation models;Tiny machine learning;Noise measurement;Acoustics;Topology;Network topology;Domain adaptation;extreme edge;keyword spotting (KWS);low-power microcontrollers;noise robustness;on-device learning (ODL);TinyML},
  doi={10.1109/JIOT.2026.3654437}}
```

```
@INPROCEEDINGS{cioflan2024ondevice,
  author={Cioflan, Cristian and Cavigelli, Lukas and Rusci, Manuele and de Prado, Miguel and Benini, Luca},
  booktitle={2024 IEEE 6th International Conference on AI Circuits and Systems (AICAS)}, 
  title={On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems}, 
  year={2024},
  volume={},
  number={},
  pages={6-10},
  keywords={Accuracy;Embedded systems;Microcontrollers;Circuits and systems;Noise;Refining;Neural networks;On-Device Learning;Domain Adaptation;Low-Power Microcontrollers;Extreme Edge;TinyML;Noise Robustness;Keyword Spotting},
  doi={10.1109/AICAS59952.2024.10595987}}

```

```
@inproceedings{cioflan2022towards,
  author={Cioflan, Cristian and Cavigelli, Lukas and Rusci, Manuele and De Prado, Miguel and Benini, Luca},
  booktitle={2022 IEEE 4th International Conference on Artificial Intelligence Circuits and Systems (AICAS)}, 
  title={Towards On-device Domain Adaptation for Noise-Robust Keyword Spotting}, 
  year={2022},
  volume={},
  number={},
  pages={82-85},
  doi={10.1109/AICAS54282.2022.9869990}}

```



## Installation

To install the packages required to run the training and adaptation of a PyTorch model, a conda environment can be created from `environment.yml` by running:
```
conda env create -f environment.yml
```
## Example

`config.json` shows a configuration example for applying ODDA on a NA-KWS pretrained DS-CNN S network on GoogleSpeechCommands v2 for the *meeting* noise from DEMAND. 

To run the main script, use the command:
```
python main.py --config_file config.json
```

## Contributor
Cristian Cioflan, ETH Zurich, [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)

## Acknowledgements
This work received support from the Swiss National Science Foundation Project 207913 "TinyTrainer: On-chip Training for TinyML devices". The work was supported in part by the Swiss State Secretariat for Education, Research, and Innovation (SERI) under the SwissChips initiative; and in part by the Horizon Europe dAIEdge Grant 101120726.

## License
The code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
