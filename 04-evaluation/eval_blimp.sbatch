#!/bin/bash
#SBATCH --job-name=BLIMP_TEST_model_name
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.chesi@iusspavia.it
#SBATCH -e "results/%x-%j.err"
#SBATCH -o "results/%x-%j.out"
#SBATCH --partition=gpu_partition
#SBATCH --chdir=/home/your_home/04-evaluation
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1

date

conda init bash
source /home/your_home/.bashrc
conda activate your_env_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

MODEL_PATH="/home/your_home/03-model_training/EN_CHI_BPE_eMG_RNN_fixed_E650_H650x1"

MODEL_BASENAME=$(basename $MODEL_PATH)

srun python -m lm_eval --model emg_lm \
    --model_args pretrained=$MODEL_PATH \
    --tasks blimp_filtered,blimp_supplement \
    --device cuda:0 \
    --log_samples \
    --batch_size 128 \
    --output_path results/blimp/${MODEL_BASENAME}/blimp_results.json

python plot_results.py blimp ${MODEL_BASENAME}

date
echo "end of job"