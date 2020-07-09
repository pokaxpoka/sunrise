for seed in 1234 2341 3412 4123; do
    python examples/sac.py --env $1 --seed $seed --num_layer 2
done
