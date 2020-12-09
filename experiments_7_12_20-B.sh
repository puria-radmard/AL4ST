# experiments_7_12_20-B.sh

# No self-supervision, so temperature does not matter
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01

# Beta = 0.01
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01

# Beta = 0.1
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01

# Beta = -0.01
python train.py -W 5 -A rand -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01

# Baseline with different normalisation (i.e. lc, mnlp, then hybrid). Therefore T and beta have no difference
python train.py -W -1 -A rand -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W -1 -A lc -T 1 -alpha 0 --beta 0 -R 10000 -I 0.01
python train.py -W -1 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W -1 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.0
python train.py -W -1 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -I 0.01