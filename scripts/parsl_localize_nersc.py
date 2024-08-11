import parsl
from parsl.app.app import bash_app
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import SlurmProvider

# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor

# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import SrunLauncher, MpiExecLauncher, GnuParallelLauncher

# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface


if __name__ == "__main__":
    run_dir = "/pscratch/sd/m/mansisak/memorization/src/localize/"
    env = "/pscratch/sd/m/mansisak/memorization/env/"

    user_opts = {
        "worker_init": f"module load conda; conda activate {env}; cd {run_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#SBATCH --constraint=gpu",  # &hbm80g",  # specify any PBS options here, like filesystems
        "account": "m1266",
        "queue": "regular",  # e.g.: "prod","debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "01:00:00",
        "nodes_per_block": 1,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        # "cpus_per_node":    32, # Up to 64 with multithreading
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
        # "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.
    }

    from parsl.config import Config
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider
    from parsl.executors import HighThroughputExecutor

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="production_grade",
                available_accelerators=[
                    "0,1,2,3"
                ],  # NOTE(MS): this is me attempting to put one process per node  # Creates 4 workers and pins one to each GPU, use only for GPU
                max_workers_per_node=1,  # NOTE(MS): this is my attempt at limited one experiment per node
                cpu_affinity="block",  # Pins distinct groups of CPUs to each worker
                provider=SlurmProvider(
                    launcher=SrunLauncher(
                        overrides="--gpus-per-node 4 -c 64"
                    ),  # Must supply GPUs and CPU per node
                    walltime="12:00:00",
                    nodes_per_block=8,  # So that we have a total of 4 nodes * 4 GPUs
                    scheduler_options="#SBATCH -C gpu&hbm80g\n#SBATCH --qos=regular\n#SBATCH --mail-user=sakarvadia@uchicago.edu",  # Switch to "-C cpu" for CPU partition
                    account=user_opts["account"],
                    worker_init="""
    module load conda
    conda activate /pscratch/sd/m/mansisak/memorization/env/
    cd /pscratch/sd/m/mansisak/memorization/src/localize/

    # Print to stdout to for easier debugging
    module list
    nvidia-smi
    which python
    hostname
    pwd""",
                ),
            )
        ]
    )

    parsl.load(config)

    @bash_app
    def localize_memorization_prod_grade(
        model_name="",
        step=143000,
        seed=0,
    ):
        exec_str = f"python localize_hp_sweep.py --model_name {model_name}  --step {step} --seed {seed}"

        return f" env | grep CUDA; {exec_str};"

    param_list = []

    for seed in [
        0,
        # NOTE(MS): we don't need multiple seeds for prod grade experiments
        # 1,
        # 3,
    ]:
        for model_name in [
            "EleutherAI/pythia-6.9b-deduped",
            "EleutherAI/pythia-2.8b-deduped",
        ]:
            for step in [36000, 72000, 108000, 143000]:
                args_dict = {
                    "model_name": f"{model_name}",
                    "step": f"{step}",
                    "seed": f"{seed}",
                }
                param_list.append(args_dict)

    print("Number of total experiments: ", len(param_list))
    futures = [localize_memorization_prod_grade(**args) for args in param_list]

    for future in futures:
        print(f"Waiting for {future}")
        print(f"Got result {future.result()}")

        # with open(future.stdout, "r") as f:
        #    print(f.read())
