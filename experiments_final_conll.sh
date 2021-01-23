conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.7 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.5 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #4
conda activate experiments && python train.py -W 3 7 -A lc -T 1 -alpha 0.3 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 0.7 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #4
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 0 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #4
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 0.5 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #4
conda activate experiments && python train.py -W '-1' -A lc -T 1 -alpha 0.3 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3

conda activate experiments && python train.py -W 3 7 -A maxent -T 1 -alpha 0.7 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A maxent -T 1 -alpha 0 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A maxent -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A maxent -T 1 -alpha 0.5 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W 3 7 -A maxent -T 1 -alpha 0.3 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A maxent -T 1 -alpha 0.7 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A maxent -T 1 -alpha 0 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A maxent -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A maxent -T 1 -alpha 0.5 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A maxent -T 1 -alpha 0.3 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3


conda activate experiments && python train.py -W 3 7 -A rand -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3
conda activate experiments && python train.py -W '-1' -A rand -T 1 -alpha 1 --beta 0 -R 2000 -B 1 -I 0.01 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3

conda activate experiments && python train.py -W -1 -A baseline -T 1 -alpha 0.7 --beta 0 -R 1 -B 1 -I 1 -epochs 50 --earlystopping 100 --data_path /home/pradmard/repos/data/CONLL/conll2003/ #3