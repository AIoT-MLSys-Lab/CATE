#!/usr/bin/env bash

for s in {1..200}
	do
		python search_methods/dngo_ls_nasbench101.py --dim 64 --seed $s --init_size 16 --topk 5 --dataset nasbench101 --output_path bo  --embedding_path cate_nasbench101.pt
	done
