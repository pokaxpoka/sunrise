# SUNRISE: OpenAI Gym Experiments

This codebase was originally forked from [rlkit](https://github.com/vitchyr/rlkit).  

## install

1. Install and use the included Ananconda environment
```
$ conda env create -f environment/linux-gpu-env.yml
$ source activate rlkit
```
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable or simply
run:
```
pip install -e .
```

3. Install ["benchmarking MBRL"](https://arxiv.org/abs/1907.02057),
```
pip uninstall gym
pip install gym==0.9.4 mujoco-py==0.5.7 termcolor
cd mbbl_envs
pip install --user -e .
```

## Run experiments

### SAC on state observarion
```
./scripts/run_sac.sh [env_name]
```

### SUNRISE on state observarion
```
./scripts/run_sunrise.sh [env_name] [beta] [temperature] [lambda]
```
