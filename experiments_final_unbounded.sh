conda activate experiments && python train.py -W 1 100 -A lc -T 1 -alpha 0.3 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/
conda activate experiments && python train.py -W 1 100 -A maxent -T 1 -alpha 0.9 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 1 100 -A lc -T 1 -alpha 0.7 --beta 0 -R 10000 -B 1 -I 0.01
conda activate experiments && python train.py -W 1 100 -A maxent -T 1 -alpha 0.5 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/