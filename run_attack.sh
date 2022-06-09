#!/bin/bash
for model in "dbpedia14" "ag_news" "imdb" "yelp"
do
    for idf in $(seq 10 10 100)
    do
        python whitebox_attack.py --data_folder data --dataset ${model} --model gpt2 --finetune True --start_index 0 --num_samples 100 --gumbel_samples 100 --top_idf_percent ${idf} > "./attack_results/${model}_${idf}.txt" 2>&1
    done
done

for model in "dbpedia14" "ag_news" "imdb" "yelp"
do
    for idf in $(seq 10 10 100)
    do
        python evaluate_adv_samples.py --data_folder data --dataset ${model} --surrogate_model gpt2 --target_model gpt2 --finetune True --start_index 0 --num_samples 100 --end_index 100 --gumbel_samples 1000 --top_idf_percent ${idf} > "./evaluate_results/${model}_${idf}.txt" 2>&1
    done
done
