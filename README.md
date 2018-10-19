# DDQnetwork
AI agent via reinforcement learning

## Overview
This repository contains code for a project that involved Q-Learning. The recommended actions and responses happen outside of the model itself. Consequently, a fair amount of code (notably Env.py and Buffer.py) are needed to structure the data into the requisite structure for the Q-Learning neural network.

### Packages
pandas, numpy, tensorflow, sklearn, matplotlib, functools, pymongo (for ingesting data)

## Navigating this repository
### code folder
*Buffer.py* - Loads the experience replay buffer class for Q-Learning

*clean_mongo.py* - functions for cleaning data used by pipe_ce and pipe_ca (not included in repo)

*constants.py* - stores constants used as parameters by other scripts (not included in repo)

*craft_synthetic_data.py* - because initially there is only the default behavior this program creates some synthetic data to ape the other behaviors so that when the program goes live the neural network won't give super wonky predictions

*eda_model.py* - the initial predictive model code to ascertain that the Q-Learning model will actually work (not included in repo)

*Env.py* - class to load cleaned mongo data into a structure ready for Buffer class

*get_card_auth.py* - bank code response data in mongo (not included in repo)

*get_collection_engine.py* - collection engine data (not included in repo)

*get_raw_data_for_scalers.py* - This script collects data from different days within a time period to create the initial # of variables as well as create/save sklearn's 
standardscaler object

*grid_search.py* - grid searching neural network hyperparameters

*helpers.py* - helper functions for other scripts

*Model.py* - child class of DoubleDuelQ class. Provides functions to train model, output model diagnostics, and output model predictions

*pipe_ca.py* - ingest/clean data from card_auth mongo collection

*pipe_ce.py* - ingest/clean data from collection_engine mongo collection

*pipe_model.py* - calls other pipe functions. Ingests raw data, cleans the data, and pipes into Env and Buffer classes

*pipe_predict_test_call.py* - calls pipe_predict_test for use in testing

*pipe_predict_test.py* - used for testing for pipe_predict

*pipe_predict.py* - pipes in data and gets prediction outputs

*Qnetwork.py* - the core neural network code

*train_initial.py* - train the model on historical data

*train.py* - similar to train_initial except it is called when model is in production
