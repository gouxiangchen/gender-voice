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
to train a model using 3 fully connected layers, and finnaly a model parameters file would be saved. It is **2.07MB** in size

and the accuracy in training set would approch nearly **100%** and **96%** in testing set.

run `python with_conv1d.py`

to train a model using conv1d with much more layers depth than the fully connected, and the **residual learning** strategy is used to handle the deeper depth training.

finnaly a model parameters file would be saved and it is **57.2MB** in size (much more than fully connected layers model)

the accuracy in training set would approch nearly **100%** and **97%** in testing set (quite sight improvement~ but it works).

