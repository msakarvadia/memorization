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

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="production_grade",
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                max_workers_per_node=1,  # NOTE (MS): I set this from 4 to 1 to ensure 1 process per node
                available_accelerators=user_opts[
                    "available_accelerators"
                ],  # if this is set, it will override other settings for max_workers if set
                # cores_per_worker=user_opts["cores_per_worker"],
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                provider=SlurmProvider(
                    launcher=SrunLauncher(
                        overrides="--gpus-per-node 4 -c 64"
                    ),  # Must supply GPUs and CPU per node
                    account=user_opts["account"],
                    qos=user_opts["queue"],
                    # select_options="ngpus=4",
                    # PBS directives (header lines): for array jobs pass '-J' option
                    scheduler_options=user_opts["scheduler_options"],
                    # Command to be run before starting a worker, such as:
                    worker_init=user_opts["worker_init"],
                    # number of compute nodes allocated for each block
                    nodes_per_block=user_opts["nodes_per_block"],
                    init_blocks=1,
                    min_blocks=0,
                    max_blocks=1,  # Can increase more to have more parallel jobs
                    # cpus_per_node=user_opts["cpus_per_node"],
                    walltime=user_opts["walltime"],
                ),
            ),
        ],
        # run_dir=run_dir,
        checkpoint_mode="task_exit",
        retries=2,
        app_cache=True,
    )

    from parsl.config import Config
    from parsl.launchers import SrunLauncher
    from parsl.providers import SlurmProvider
    from parsl.executors import HighThroughputExecutor

    config = Config(
        executors=[
            HighThroughputExecutor(
                available_accelerators=4,  # Creates 4 workers and pins one to each GPU, use only for GPU
                cpu_affinity="block",  # Pins distinct groups of CPUs to each worker
                provider=SlurmProvider(
                    launcher=SrunLauncher(
                        overrides="--gpus-per-node 4 -c 64"
                    ),  # Must supply GPUs and CPU per node
                    walltime="01:00:00",
                    nodes_per_block=1,  # So that we have a total of 8 GPUs
                    scheduler_options="#SBATCH -C gpu\n#SBATCH --qos=regular",  # Switch to "-C cpu" for CPU partition
                    account=user_opts["account"],
                    worker_init="""
    module load conda
    conda activate /pscratch/sd/m/mansisak/memorization/env/
    cd /pscratch/sd/m/mansisak/memorization/env/

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
    def localize_memorization(
        batch_size=128,
        lr=1e-3,
        data_name="increment",
        num_7=20000,
        num_extra_data=3000,
        epochs=4000,
        seed=0,
        length=20,
        max_ctx=150,
        n_layers=1,
        dup=0,
        backdoor=0,
        ckpt_epoch=100,
    ):

        print("The seed for this experiment is: ", seed)
        exec_str = "echo hi"

        return f" env | grep CUDA; {exec_str};"

    param_list = []

    for seed in [
        0,
        1,
        2,
        # 3,
        # 4,
    ]:
        for lr in [1e-3]:
            for batch_size in [128]:
                for extra_data_size in [3000, 10000, 20000]:
                    for dup in [0, 1]:
                        for backdoor in [1, 0]:
                            for data_name in ["mult", "increment", "wiki_fast"]:
                                for layer in [2, 4, 8, 16]:
                                    if (
                                        data_name in ["mult", "increment"]
                                        and not backdoor
                                    ):
                                        epochs = [500, 1500, 2500, 3500]
                                    if data_name in ["mult", "increment"] and backdoor:
                                        epochs = [50, 200, 350, 500]
                                    if data_name in ["wiki_fast"] and not backdoor:
                                        epochs = [10, 40, 70, 100]
                                    if data_name in ["wiki_fast"] and backdoor:
                                        epochs = [10, 20, 30, 50]
                                    for ckpt_epoch in epochs:

                                        # for language data, we only want to iterate once (not for each extra data size)
                                        if (
                                            data_name == "wiki_fast"
                                            and extra_data_size != 20000
                                        ):
                                            continue
                                        # we only want to train language on duplicated data
                                        if data_name == "wiki_fast" and dup == 0:
                                            continue
                                        if data_name == "mult" and dup == 1:
                                            continue
                                        if data_name == "increment" and dup == 1:
                                            continue

                                        args_dict = {
                                            "n_layers": f"{layer}",
                                            "batch_size": f"{batch_size}",
                                            "lr": f"{lr}",
                                            "data_name": f"{data_name}",
                                            "num_7": f"20000",
                                            "num_extra_data": f"{extra_data_size}",
                                            "epochs": f"3500",
                                            "seed": f"{seed}",
                                            "length": f"20",
                                            "max_ctx": f"150",
                                            "dup": f"{dup}",
                                            "backdoor": f"{backdoor}",
                                            "ckpt_epoch": f"{ckpt_epoch}",
                                        }
                                        param_list.append(args_dict)

    print("Number of total experiments: ", len(param_list))
    futures = [localize_memorization(**args) for args in param_list]

    for future in futures:
        print(f"Waiting for {future}")
        print(f"Got result {future.result()}")

        # with open(future.stdout, "r") as f:
        #    print(f.read())
