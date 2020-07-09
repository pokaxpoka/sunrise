for seed in 1234 2341 3412 4123; do
        python examples/sunrise.py --env $1 --seed $seed --num_layer 2 --num_ensemble 5 --ber_mean $2 --temperature $3  --inference_type $4 
done
