# gender-voice
gender recognize using pre-processed voice data by deep neural networks

## Requirements 

- tensorflow (for tensorboard logging)
- pytorch (>=1.0, 1.0.1 used in my experiment)
- pandas and other common python packages

## Dataset
from [kaggle](https://www.kaggle.com/primaryobjects/voicegender) and excel files are also attached in this repo

## Usage

This repo contains two structures of neural networks to recognize gender of pre-processed voice data (2-classification problem)

run `python with_fully_connect.py`
to train a model using 3 fully connected layers, and finally a model parameters file would be saved. It is **2.07MB** in size

and the accuracy in training set would approch nearly **100%** and **96%** in testing set.

run `python with_conv1d.py` to train a model using conv1d with much more layers depth than the fully connected, and the **residual learning** strategy is used to handle the deeper depth training.

finally a model parameters file would be saved and it is **57.2MB** in size (much bigger than fully connected layers model)

the accuracy in training set would approch nearly **100%** and **97%** in testing set (quite slight improvement~ but it works).

**AFTER TRAINING**, you can run `python test_model.py -f` and `python test_model.py -c` to test the trained fully connected model and conv1d model in test set respectively, 
and it would produce a txt file contains the female probabilities of test data per row.
