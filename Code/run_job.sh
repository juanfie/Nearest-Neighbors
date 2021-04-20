#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -j y
#$ -V
#$ -m e
#$ -M rafaelcedenog@gmail.com
#$ -N CENACE_FORECASTING
export PATH=$PATH:$HOME/anaconda3/bin
source activate nnenv

FILE_PATH=$1
SERIES_NAME=$2
python cenace.py $FILE_PATH $SERIES_NAME


