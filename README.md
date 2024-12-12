# TrackFormers-Models

TrackFormers is a machine learning framework for track reconstruction in particle physics experiments. It leverages transformer- and U-Net-inspired deep learning architectures to predict particle tracks from hit data.

This repository contains 4 submodules corresponding to the 4 models described in the paper [TrackFormers: In Search of Transformer-Based Particle Tracking for the High-Luminosity LHC Era](https://arxiv.org/abs/2407.07179). EncDec, EncCla, and EncReg are transformer-based models, whereas U-Net is, as the name suggests, a U-Net model.

## Table of Contents
- [Models](#models)
- [Usage](#usage)
- [Citation](#citation)
- [Authors](#authors)
- [Licence](#licence)

## Models
- [EncReg](#EncReg)
- [EncDec](#EncDec)
- [EncCla](#EncCla)
- [U-Net](#U-Net)

## EncReg
This submodule contains the implementation of the Transformer Regressor (EncReg): an encoder-only model for track finding. It is a sequence-to-sequence model, taking a sequence of hits from a single event as input, and producing a sequence of track parameters: one per hit, characterizing the track of the particle that generated each hit. In a secondary step, a clustering algorithm is used to group hits belonging to the same particle together (HDBSCAN).

The submodule implements the whole pipeline, from data loading, to training, to evaluation. The scoring of the model is done using the TrackML score, which is taken from the trackML github page (https://github.com/LAL/trackml-library/tree/master), and three efficiency metrics defined in the GNN Tracking project (https://github.com/gnn-tracking/gnn_tracking/tree/main).

### Dependencies
The code runs on Python==3.9.7 and uses torch==2.1.2.

Other libraries used are as follows: numpy, pandas, matplotlib, hdbscan.

### Contents
The Transformer Regressor architecture is implemented in `model.py`, and the Flash attention and custom encoder layer using it are implemented in `custom_encoder.py`. The `training.py` file contains the training functionality when exact attention is used, while `training_flash.py` contains training funcitonality with Flash attention enabled. 

The `evaluation\` directory contains functionality used at inference time: `test.py` for obtaining scores and heatmaps; `performance_stats.py` for obtaining MSE, CUDA and CPU time, standard deviation; `plotting.py` for the creation of the heatmaps; `scoring.py` for the TrackML and efficiency calculations.

The `data_processing\` directory contains functionality related to the simulated data: `dataset.py` for loading REDVID data, and a data class for the hits and their associated particle IDs; `trackml_data.py` for loading the TrackML data and transforming the TrackML events into smaller ones for the creation of the subset datasets. The `domain_decomposition.py` functionality is not used in the reported experiments but is fully functional. 

The `refiner\` folder contains the implementation of a refiner network (training, testing), which is not used in the reported experiments but is also fully functional.

The trained models for which best scores are reported in the paper, are included in the `models\` folder.

### Using the Code Base
To train a model, simply run the `training.py` file and provide it with the commandline arguments it expects: `max_nr_hits, nr_epochs, data_path, model_name, nr_enc_layers, embedding_size, hidden_dim, data_type, dropout, early_stop`. Some have a set default value that can be see in the `training.py.` file. Alernatively, you can run the `training_flash.py` file, which expects the same arguments, but also makes use of Flash Attention instead of the default Multi-Head Attention.

To evaluate a model using the TrackML score, simply run the `test.py` file from the `evaluation\` directory and provide it with the commandline arguments it expects: `max_nr_hits, data_path, model_name, nr_enc_layers, embedding_size, hidden_dim, data_type, dropout`. Alternativelly, to also obtain the three additional efficiency metrics and the timing information (CPU and GPU time) of running the model, run the `performance_stats.py` file from the same directory. It expects the same arguments.

Example usage can be found in `script_train.sh` and `evaluation\script_test.sh`.

## EncCla
The encoder-only classifier (EncCla) outputs a class label for each hit in an event, given pre-defined bins as class_ids in the track parameter space. It takes a sequence of all hits from a single event as input, and outputs a sequence of class labels.

### Usage
Running training:
```
python train.py <path_to_toml_file>
```
Running evaluation:
```
python evaluate.py <path_to_toml_file>
```

## EncDec
The encoder-decoder model (EncDec) takes initial hits of a track as input, conditioned on the entire set of hits in a single event, and outputs a sequence of hits for each track.

### Installation

1. Clone the repository:
    ```sh
    git clone <repo_here>
    cd trackformers
    ```

2. Set up the environment:
    ```sh
    conda env create -f environment.yml
    conda activate tracking-gpu
    ```

### Usage
#### Data Preparation
To prepare data for training, use either the prepare_parsed_data_trackml_memeff.ipynb or the prepare_parsed_data_redvid_memeff.ipynb notebook. These notebooks convert data from the trackml format to a format that the framework can use.

#### Training

To train the model, use the `train_wandb.py` script, which trains the model and uploads results to the weights and biases platform. You can specify various parameters such as the number of tracks, maximum size of tracks, and input file name.

Example:
```sh
python train_wandb.py --number_of_tracks=1000 --max_size_tracks=30 --max_size_hits=450 --infile_name=data.csv --parsed_data_dir=./parsed_data --save_dir=saved_models/training_temp
```

#### Evaluation:

To evaluate the model, use the eval_result.py script. This script will load the trained model and evaluate its performance on the dataset.


Example:
```sh
python eval_result.py --number_of_tracks=1000 --max_size_tracks=30 --max_size_hits=450 --file_name=data.csv --load_dir=saved_models/training_temp
```

#### Plotting Results
To generate and save a 3D plot of the decoded and true tracks, use the plot_result.py script.

Example:
```sh
python plot_result.py --number_of_tracks=1000 --max_size_tracks=30 --max_size_hits=450 --file_name=data.csv --load_dir=saved_models/training_temp --plotted_track=0 --output_dir=results
```

## U-Net
This repository contains the implementation of the U-Net model. The files are described as follows.

- **models.py**. This script contains the main for the methodologies described as the U-Net classifier (the one used during the original manuscript) and a U-Net regressor, written under the classes 'UNet_SCN_classifier' and 'UNet_SCN_regressor', respectively. As explained in the paper, sparse convolutions are used with the aid of [spconv](https://github.com/traveller59/spconv) module, written in PyTorch. Everything is accurately documented along the entire script.

- **utils.py**. This file contains several utilities such as custom losses that can be considered, and the code used to compute the original *TrackML* score. Again, each function has its own documentation where being defined. The interested reader is addressed to them.

- **training_script.py**. Example code to run a training procedure.
	- **lines 1-28**. Import modules and define particular parameters such as the device used for the training process, the DTYPE of the tensors and the batch size.
	- **lines 31-75**. Load the data from an external *.h5* file that contains the indices for the training, validation and test sets. These indices are crucial to construct the actual sparse tensors that will be fed into the U-Net.
	- **lines 77-108**. The sparse tensors are constructed as pairs (X,Y) for the three datasets of interest.
	- **lines 112-121**. Define the model as the 'UNet_SCN_classifier' and different parameters such as the learning rate of the optimizer and the number of bins within the data (i.e., 'n1' and 'n2').
	- **lines 123-165**. Finally, the training sloop is written. A lot of information is saved here such as the accuracy and *trackML* score for all datasets and the losses vs. number of epochs.

### Requirements

The entire code is written using PyTorch and fully tested with Python 3.8.10. Main used packages are:

* PyTorch
* spconv
* NumPy
* Matplotlib

Other minor packages such as *h5py* are used. For further research about the compatibilities between different versions of Python and PyTorch for the *spconv* package, the reader is addressed to their own repository [here](https://github.com/traveller59/spconv).


## Usage
### Clone the whole respository

## Clone a single submodule
- Copy the submodule's URL from the .gitmodules file or GitHub
- Clone the submodule using its URL:
  ```sh
  git clone https://github.com/Submodule_URL.git
  ```

## Citation
If you use this codebase in your research or publication, we kindly request you to 
cite the following paper:

```bibtex
@article{Caron:YEAR:TrackFormers,
  author = {Caron, Sascha and Dobreva, Nadezhda and Ferrer Sánchez, Antonio and 
    Martín-Guerrero, José D. and Odyurt, Uraz and Ruiz de Austri Bazan, Roberto and 
    Wolffs, Zef and Zhao, Yue},
  title = {TrackFormers: In Search of Transformer-Based Particle Tracking for the 
    High-Luminosity LHC Era}, 
  journal = {UPDATE AS APPROPRIATE},
  year = {UPDATE AS APPROPRIATE},
  doi = {GIVEN IN ZENODO METADATA}
}
```

## Authors
The codebase is created by:
- EncDec: _Nadezhda Dobreva_ - Radboud University, the Netherlands
- EncCla: _Yue Zhao_ - SURF, the Netherlands
- EncReg: _Zef Wolffs_ - University of Amsterdam; Nikhef, the Netherlands
- U-Net: _Antonio Ferrer Sánchez_ - University of Valencia, Spain

The collaborating team of the paper includes:
- _Sascha Caron_ - Radboud University; Nikhef, the Netherlands
- _Uraz Odyurt_ - University of Twente; Nikhef, the Netherlands
- _José D. Martín-Guerrero_ - University of Valencia, Spain
- _Roberto Ruiz de Aurtri Bazan_ - University of Valencia, Spain


## License
This project is licensed under the MIT License. See the LICENSE file for details.
