# Project Repository: spam

This is the code base for the final project of the group **spam** in the course Advanced Natural Language Processing during the Wintersemester 2022/2023.  The project members are Manh Linh Phan, Stanley Joel Gona and René Wolf.
The project's goal is to provide a viable approach to the [SemEval 2023 clickbait challenge](https://pan.webis.de/semeval23/pan23-web/clickbait-challenge.html) Subtask 1: Spoiler Classification.

### Data
The data that can be found in the repository is provided by *Matthias Hagen, Maik Fröbe, Artur Jurk, & Martin Potthast. (2022). Webis Clickbait Spoiling Corpus 2022 (1.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.6362726* and licensed under the Creative Commons attribution 4.0 International (CC BY 4.0) License.

## Models
This repository consits of three different machine learning approaches to the task, one for each member of the group:
- A Long Short-Term Memory (LSTM) model, that takes words as input
- another LSTM model that takes characters as input
- and a collection of three BERT-based transformer models.

### How to run the transformers
 - All necessary functions and libraries are defined in /transformers_pre_trained/training_pipeline.ipynb
 - Each transformer model can be re-trained and evaluated using their respective notebook file:
    + DistilBERT: **distilbert_postText_targetTitle.ipynb**
    + BERT: **bert_postText_targetTitle.ipynb**
    + DeBERTa : **deberta_postText_targetTitle.ipynb**
 - Hyper-parameters can be adjusted in the second cell of the individual transformer's notebook file
 - Start the training and evaluation by running the third cell, which contains the function train_eval_pipeline
 - The results will be output to a DataFrame - _dataframe_

### How to run the character-based LSTM
- All necessary functions are defined in four different .py-files
- The *main.py* file is the file to create and test the model.
- run *python main.py* from the command line to get the model with the best results.
- the package torch needs to be installed on your system in order to do so
- Hyperparameters can be adjustet at the beginning of the *main.py* file

### How to run the word-based LSTM
- clone the .ipynb file to the local machine
- open the notebook in Google Colab or Jupyter
- run the cells in the order they appear
