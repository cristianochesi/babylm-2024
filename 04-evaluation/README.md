## Model evaluation

This folder contains the evaluation scripts used for [BabyLM - evaluation-pipeline-2024](https://github.com/babylm/evaluation-pipeline-2024) (10M dataset - strict-small track) 
the results of the **eMG_RNN_base** trained model are in the `results` subfolder (and also available at: [NeTS-lab/eMG_RNN_base](https://huggingface.co/NeTS-lab/eMG_RNN_base)).

### Assess the model

In order to assess this custom RNN model, you need to place in the

and add to the `__init__.py` initialization file in the `lm_eval/models` folder the eMG_RNN_base architecture:

```
from . import (
    anthropic_llms,
    huggingface,
    eMG_RNN_base,
    ...
    )
```
