# On-device Domain Adaptation for Noise-Robust Keyword Spotting

## Introduction

On-device Domain Adaptation (ODDA) for Noise-Robust Keyword Spotting is a methodology aimed at increasing the robustness to unseen noises for a keyword spotting system. The objective of keyword spotting (KWS) is to detect a set of predefined keywords within a stream of user utterances. The difficulty of the task increases in real environments with significant noise. To improve the performance of a KWS system in noise conditions unseen during training, we propose a methodology for tailoring a model to on-site noises through ODDA.

### Citing

If you use our methodology in an academic context, please cite the following publication:

Publications:
* *Towards On-device Domain Adaptation for Noise-Robust Keyword Spotting* [IEEE AICAS](https://ieeexplore.ieee.org/document/9869990)
* *On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems* [arXiv preprint](https://arxiv.org/abs/2403.10549)

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

```
@misc{cioflan2024ondevice,
      title={On-Device Domain Learning for Keyword Spotting on Low-Power Extreme Edge Embedded Systems}, 
      author={Cristian Cioflan and Lukas Cavigelli and Manuele Rusci and Miguel de Prado and Luca Benini},
      year={2024},
      eprint={2403.10549},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

## Installation

To install the packages required to run the training and adaptation of a PyTorch model, a conda environment can be created from `environment.yml` by running:
```
conda env create -f environment.yml
```
## Example

`config.json` shows a configuration example for (pre)training a NL-KWS DS-CNN S network on GoogleSpeechCommands v2. 

To run the main script, use the command:
```
python main.py --config_file config.json
```

## Contributor
Cristian Cioflan, ETH Zurich, [cioflanc@iis.ee.ethz.ch](cioflanc@iis.ee.ethz.ch)


## License
The code is released under Apache 2.0, see the LICENSE file in the root of this repository for details.
