# fp_spam

This is the code base for the final project of the group spam in the course Advanced Natural Language Processing. 

## Phan
### How to run transformers
 - Every necessary functions and libraries are defined in /transformers_pre_trained/training_pipeline.ipynb
 - Each transformer model can be re-trained using their respective notebook file:
    + DistilBERT: **distilbert_postText_targetTitle.ipynb**
    + BERT: **bert_postText_targetTitle.ipynb**
    + DeBERTa : **deberta_postText_targetTitle.ipynb**
 - Hyper-parameters can be adjusted in the second cell of the individual transformer's notebook file
 - Start the training and evaluation by running the third cell, which contains the function train_eval_pipeline
 - The results will be output to a DataFrame call _dataframe_
