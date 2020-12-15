# Sweep over beta and alpha, not changing T for now
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0.1 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0.01 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta -0.01 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta -0.01 -R 10000 -B 1 -I 0.01

# Sweep over alpha and B, not changing beta
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 4 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -B 4 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -B 4 -I 0.01

# Random baseline
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -B 1 -I 0.01