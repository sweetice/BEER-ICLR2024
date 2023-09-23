
# Reproduce BEER results

The present anonymous repository serves as a guide for reproducing the results of the "BEER" method proposed in our ICLR submission "Adaptive Regularization of Representation Rank as an Implicit Constraint of Bellman Equation". 

The file tree structure of this repository is as follows:

```bash
.
├── BEER.py
├── README.md
├── conda_env.yml
├── main.py
└── utils.py

0 directories, 5 files

```


# Reproduce BEER locally

To reproduce the results locally, we recommend the following steps:

1. Create the Conda environment specified in the `conda_env.yml` file by running the command conda env create -f conda_env.yml and activate it with conda activate beer.

```
conda env create -f conda_env.yml
conda activate beer
```

2. Follow the instructions provided in the [MuJoCO](https://github.com/openai/mujoco-py), [DMControl](https://github.com/deepmind/dm_control), and [dmc2gym](https://github.com/denisyarats/dmc2gym) repositories to install the necessary dependencies.

3. Execute the following command to reproduce BEER result. 

```bash
python3 main.py --domain_name=humanoid --task_name=stand --seed=0
```


Logs of the execution will be displayed, and any issues with the environment should be addressed by consulting the documentation for DMControl and MuJoCO.



# Logs

If we run BEER code, we can see logs that look like:


```bash
(beer) python3 main.py --domain_name=humanoid --task_name=stand --seed=0
  1%|          | 10037/1000000 [00:15<3:32:11, 77.76it/s]---------------------------------------
Current Timesteps: 10000, Algo: BEER, Domain: humanoid, Task: stand, seed: 0, Evaluation over 10 episodes: 5.235
  2%|▏         | 19849/1000000 [00:19<05:23, 100.42it/s]---------------------------------------
Current Timesteps: 20000, Algo: BEER, Domain: humanoid, Task: stand, seed: 0, Evaluation over 10 episodes: 10.235
```
