#!/bin/bash
#SBATCH --nodes=3           #Numero de Nós

#SBATCH --ntasks-per-node=1 #Numero de tarefas por Nó
#SBATCH --cpus-per-task=24
#SBATCH --mem=60GB
#SBATCH -p nvidia_dev       #Fila (partition) a ser utilizada
#SBATCH -J Rec_test         #Nome job
#SBATCH --exclusive         #Utilização exclusiva dos nós durante a execução do job
#SBATCH --output=logs/slurm/slurm-%A.out

#Exibe os nos alocados para o Job
echo $SLURM_JOB_NODELIST
nodeset -e $SLURM_JOB_NODELIST

# cd $SLURM_SUBMIT_DIR/calibrated_recommendation/

#acessa o diretório onde o script está localizado
cd /scratch/calibrec/diego.silva/calibrated_recommendation/

#module load anaconda3/2020.11

recommenders=(SVD)
folds=(1 2 3)
dataset="Movielens-25M"

for i in "${recommenders[@]}";
do
    for j in 1 2 3;
    do
        echo "Recommender Job: $i"
        echo "Fold: $j"
        srun  -N 1 -n 1  /scratch/calibrec/diego.silva/.conda/envs/calibrated_recommendation/bin/python3.7 recommenders.py --recommender="$i" --fold=$j --dataset="$dataset" &
    done
done

wait