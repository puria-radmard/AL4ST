# Baseline with different normalisation (i.e. lc, mnlp, then hybrid). Therefore T and beta have no difference
python train.py -W -1 -A rand -T 1 -alpha 1 --beta 0
python train.py -W -1 -A lc -T 1 -alpha 0 --beta 0
python train.py -W -1 -A lc -T 1 -alpha 1 --beta 0
python train.py -W -1 -A lc -T 1 -alpha 0.7 --beta 0