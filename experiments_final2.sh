# Get second round of old ones
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A rand -T 1 -alpha 1 --beta 0 -R 10000 -B 1 -I 0.01

# Limit
conda activate experiments && python train.py -W -1 -A baseline -T 1 -alpha 0.7 --beta 0 -R 1 -B 1 -I 1 -num_epochs 50 -earlystopping 100

# Sweep over alpha with bigger roundsize
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.2 --beta 0 -R 50000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.3 --beta 0 -R 50000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.5 --beta 0 -R 50000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 50000 -B 1 -I 0.01
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 0.7 --beta 0 -R 50000 -B 1 -I 0.01

# Beam 5 repeat
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 5 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -B 5 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -B 5 -I 0.01

