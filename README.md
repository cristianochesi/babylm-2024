# BabyLM 2024 shared task @ NeTS 

In repository contains the data related to our participation to the [BabyLM 2024](https://babylm.github.io/) competition.

### 01-preprocess
This folder contains our preprocessing procedure. 
We decided to minimally clean the data source by removing any metalinguistic 

### 02-tokenization
You will find here an original tokenizer, dubbed **MorPiece** (**MoP**) freely based on [Tolerance Principle](https://lingbuzz.net/lingbuzz/004146) by Charles Yang.
The current version (v.0.0.1) will be soon updated. We didn't use this tokenizer in the English experiments for lack of time, but this seems to us a useful contribution for rich morphological languages.

### 03-model_training
Here we include the revisions of the standard Recurrent Neural Networks versions (based on **GRU** and **LSTM** models) we adopted in this study. These are lazily inspired by certain (non-standard) processing interpretations of Minimalist Grammars ([e-MGs](https://github.com/cristianochesi/e-MGs)) with the overt intent to model various bias in training (using specific gates combinations) that should mimic standard constraints that are operative in structure building like [C-command](http://www.glottopedia.org/index.php/C-command) and [locality](http://glottopedia.org/index.php/Locality).

### 04-evaluation
Results of the lm-eval campaign for BabyLM 2024 are included here in .json format (BLiMP task).
