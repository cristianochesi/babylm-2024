#!/bin/bash
#SBATCH --job-name=BabyLM_2024_corpus_cleaning
#SBATCH --mail-type=ALL
#SBATCH --mail-user=cristiano.chesi@iusspavia.it
#SBATCH -e "results/%x-%j.err"
#SBATCH -o "results/%x-%j.out"
##SBATCH --partition=cpuq
##SBATCH --nodelist=cn01
##SBATCH --nodelist=gp01
#SBATCH --chdir=/your_dir/01-preprocess
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --ntasks-per-core=1
##SBATCH --ntasks=1
#SBATCH --nodes=1
##SBATCH --gpus-per-node=4
##SBATCH --mem=100000M

date

conda init bash
source /your_dir/.bashrc
conda activate your_pythorch_env_for_gpu
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

DATA_DIR_SRC="../data/eng/"
DATA_DIR_PROCESSED="../data/eng/processed/"

mkdir -p $DATA_DIR_PROCESSED

python childes.py $DATA_DIR_SRC"childes/" $DATA_DIR_PROCESSED
python subtitles.py $DATA_DIR_SRC"subtitles/" $DATA_DIR_PROCESSED
python conversations.py $DATA_DIR_SRC"conversations/" $DATA_DIR_PROCESSED
python gutenberg.py $DATA_DIR_SRC"gutenberg/" $DATA_DIR_PROCESSED
python wikipedia.py $DATA_DIR_SRC"wikipedia/" $DATA_DIR_PROCESSED

cat $DATA_DIR_PROCESSED"childes.txt" $DATA_DIR_PROCESSED"subtitles.txt" $DATA_DIR_PROCESSED"conversations.txt" $DATA_DIR_PROCESSED"gutenberg.txt" $DATA_DIR_PROCESSED"wikipedia.txt" > $DATA_DIR_PROCESSED"all.txt"

python corpus_info.py $DATA_DIR_PROCESSED"all.txt"

#python segment.py

date
echo "end of job"