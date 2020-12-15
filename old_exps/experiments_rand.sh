# Random fixed window size selection (hence normalisation does not matter)
# No self-supervision, so temperature does not matter
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0

# Beta = 0.01
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.01
python train.py -W 5 -A rand -T 2 -alpha 1 --beta 0.01
python train.py -W 5 -A rand -T 3 -alpha 1 --beta 0.01

# Beta = 0.1
python train.py -W 5 -A rand -T 1 -alpha 1 --beta 0.1
python train.py -W 5 -A rand -T 2 -alpha 1 --beta 0.1
python train.py -W 5 -A rand -T 3 -alpha 1 --beta 0.1

# Beta = -0.01
python train.py -W 5 -A rand -T 1 -alpha 1 --beta '-0.01'
python train.py -W 5 -A rand -T 2 -alpha 1 --beta '-0.01'
python train.py -W 5 -A rand -T 3 -alpha 1 --beta '-0.01'