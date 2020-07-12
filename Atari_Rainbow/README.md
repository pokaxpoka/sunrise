# SUNRISE: Atari Experiments

This codebase was originally forked from [Kaixhin/Rainbow](https://github.com/Kaixhin/Rainbow).  

## install

To install all dependencies with Anaconda run `conda env create -f environment.yml` and use `source activate rainbow` to activate the environment.


## Run experiments

### Rainbow on Atari
```
./scripts/run_rainbow.sh [env_name]
```

### SUNRISE on Atari
```
./scripts/run_sunrise.sh [env_name] [beta] [temperature] [lambda]
```