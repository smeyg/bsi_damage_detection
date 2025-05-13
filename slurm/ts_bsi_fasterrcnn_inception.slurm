#!/bin/bash 
#SBATCH -p akya-cuda 
#SBATCH -A iuzun 
#SBATCH -J bsi_od_proc 
#SBATCH -N 1 # number of compute nodes
#SBATCH -n 20 # numer of cores to be used 
#SBATCH --gres=gpu:2  # number of GPUs to be assigned
#SBATCH --time=20:00:00 # max execution time
#SBATCH –-mail-type=ALL 
#SBATCH –-mail-user=uzunismail2016@gmail.com 
#SBATCH --workdir=/truba_scratch/iuzun 

now=$(date +"%T") 
echo "Start time : $now" 
source activate tensorflow_gpu1 
module load centos7.3/lib/cuda/10.0 

cd /truba/home/iuzun/ts_bsi_inception_gpu1/models/research/object_detection 
python model_main.py --logtostderr --model_dir=training/ pipeline_config_path=training/faster_rcnn_inception.config 

now=$(date +"%T") 
echo "End time : $now" 
sstat --j $SLURM_JOB_ID # isin o ana kadar kullanmis oldugu sistem kaynaklari 
echo "SLURM_NODELIST $SLURM_NODELIST" 
echo "NUMBER OF CORES $SLURM_NTASKS" 
exit