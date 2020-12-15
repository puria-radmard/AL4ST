# new_experiments_7_12_20.sh

# Fixed window size
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 5 -A lc -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 5 -A lc -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01

# variable window size
python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01
python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01

# Beam search - for later
# python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0 -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0.01 -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta 0.01 -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta 0.1 -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta 0.1 -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 1 -alpha 1 --beta '-0.01' -R 10000 -I 0.01 -B 3
# python train.py -W 2 7 -A lc -T 2.5 -alpha 1 --beta '-0.01' -R 10000 -I 0.01 -B 3