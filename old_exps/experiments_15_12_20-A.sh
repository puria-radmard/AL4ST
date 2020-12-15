# new_experiments_7_12_20.sh

# No self-supervision, so temperature does not matter
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01

# Beta = 0.01
#python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
#python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01

# Beta = 0.1
#python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
#python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01

# Beta = -0.01
#python train.py -W 5 -A rand -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
#python train.py -W 5 -A rand -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01