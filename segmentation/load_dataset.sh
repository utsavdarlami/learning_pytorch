#!/bin/bash

files=("train.zip"
       "train_hq.zip"
       "train_masks.zip"
       "train_masks.csv.zip")

for file in ${files[*]}
do
    echo $file
    # kaggle competitions download -c carvana-image-masking-challenge -f $file
done    
