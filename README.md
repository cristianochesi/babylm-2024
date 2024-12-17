# BabyLM 2024 shared task @ NeTS 

In repository contains the data related to our participation to the [BabyLM 2024](https://babylm.github.io/) competition.

### 01-preprocess
This folder contains our preprocessing procedure. 
We decided to minimally clean the data source by removing any metalinguistic 

### 02-tokenization
You will find here an original tokenizer, dubbed **MorPiece** (**MoP**) freely inspired to the [Tolerance Principle](https://lingbuzz.net/lingbuzz/004146) by Charles Yang.
The current version (v.0.0.1) will be soon updated. We didn't use this tokenizer in the English experiments for lack of time, but this seems to us a useful contribution for rich morphological languages.

### 03-model_training
Here we include our original elaboration of some standard Recurrent Neural Network architectures (**GRU** and **LSTM** models). These models are loosely inspired by certain (non-standard) processing interpretations of Minimalist Grammars ([e-MGs](https://github.com/cristianochesi/e-MGs)) with the specific intent of modeling various biases in training (using specific gate combinations) that mimic standard constraints operative in structure building, such as [C-command](http://www.glottopedia.org/index.php/C-command) and [locality](http://glottopedia.org/index.php/Locality).

### 04-evaluation
Results of the lm-eval campaign for BabyLM 2024 are included here in .json format (BLiMP task).

### REFERENCE

[Chesi et al. 2024 - Different Ways to Forget: Linguistic Gates in Recurrent Neural Networks](https://github.com/cristianochesi/babylm-2024/blob/main/chesi%20et%20al%202024%20-%20two%20ways%20to%20forget%20-%20BabyLM%202024.pdf)
