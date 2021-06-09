#!/usr/bin/env bash

for s in {1..200}
	do
		python search_methods/dngo_ls_nasbench301.py --dim 64 --seed $s --init_size 16 --topk 5 --dataset nasbench301 --output_path bo  --embedding_path cate_nasbench301.pt
	done
