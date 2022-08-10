#!/bin/bash

# knowbert
git clone git@github.com:allenai/kb.git ./models/knowbert/source
wget https://allennlp.s3-us-west-2.amazonaws.com/knowbert/models/knowbert_wordnet_model.tar.gz -O ./models/knowbert/knowbert_wordnet_model.tar.gz


# OpenEntity
wget http://nlp.cs.washington.edu/entity_type/data/ultrafine_acl18.tar.gz -O ./downstream_tasks/data/open_entity.tar.gz