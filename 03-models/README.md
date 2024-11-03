## Model architecture and training

This folder contains the model structures trained for the BabyLM 2024 competition (10M dataset - strict-small track) 
the **eMG_RNN_base** trained model is available here: [NeTS-lab/eMG_RNN_base](https://huggingface.co/NeTS-lab/eMG_RNN_base)). We also include some improvement to test different training regimen strategies (naturalistic, conversational or redundant) as described in the original paper (Chesi et al. 2024, Fusco et al. 2024)

### Train the model from scratch

To train a model simply run 

```bash
python ./train_eMG_RNN_BPE.py [training_corpus] [embedding dimension] [hidden dimension] [number of recurrent hidden layers] [regimen]
```
we strongly recommend to train the model on a high performance cluster with a decent number of GPUs (each iteraction with 2 dedicated A100 NVIDIA GPU dedicated, requires about **8h** per training epoch using **redundant** regimen with **seq_length=74**)

e.g.
```bash
#!/bin/bash
#SBATCH --job-name=eMG_RNN_training
#SBATCH --partition=your_partition
#SBATCH --nodelist=your_gpu_node
#SBATCH --chdir=/your_home/
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-core=1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2

conda init bash
source /home/your_environment/.bashrc
conda activate your_environment
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

srun python model_eMG_RNN.py ../data/eng/processed/all_eng.txt 650 650 1 conversational
```
Required python packages (to be installed before running the script):

 - `smart-open`
 - `torch`
 - `transformers`
 - `tokenizers`
 - `tqdm`