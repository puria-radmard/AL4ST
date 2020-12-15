# LC fixed window size selection (hence normalisation does not matter)
# No self-supervision, so temperature does not matter
python train.py -W 5 -A lc --beta 0 -T 1 -alpha 1

# Beta = 0.01
python train.py -W 5 -A lc --beta 0.01 -T 1 -alpha 1
python train.py -W 5 -A lc --beta 0.01 -T 2 -alpha 1
python train.py -W 5 -A lc --beta 0.01 -T 3 -alpha 1

# Beta = 0.1
python train.py -W 5 -A lc --beta 0.1 -T 1 -alpha 1
python train.py -W 5 -A lc --beta 0.1 -T 2 -alpha 1
python train.py -W 5 -A lc --beta 0.1 -T 3 -alpha 1

# Beta = -0.01
python train.py -W 5 -A lc --beta '-0.01' -T 1 -alpha 1
python train.py -W 5 -A lc --beta '-0.01' -T 2 -alpha 1
python train.py -W 5 -A lc --beta '-0.01' -T 3 -alpha 1