#!/bin/bash


git clone git@github.com:allenai/kb.git ./models/knowbert/source
wget https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wordnet_model.tar.gz -O ./models/knowbert/knowbert_wordnet_model.tar.gz