for seed in 123 231 312; do
    python sunrise.py --game $1 --num-ensemble 5 --seed $seed --target-update 2000 --T-max 500000 --learn-start 1600 --memory-capacity 500000 --replay-frequency 1 --multi-step 20 --architecture data-efficient --hidden-size 256 --learning-rate 0.0001 --evaluation-interval 10000 --id sunrise --beta-mean $2 --temperature $3 --ucb-infer $4
done
