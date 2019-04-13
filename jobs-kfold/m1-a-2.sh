#PBS -l walltime=24:00:00
cd ~/sperm
./train-kfold.py m1 a 5000 0 2
