import parsl
from parsl.app.app import bash_app
from parsl.config import Config

# PBSPro is the right provider for Polaris:
from parsl.providers import PBSProProvider

# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor

# You can use the MPI launcher, but may want the Gnu Parallel launcher, see below
from parsl.launchers import MpiExecLauncher, GnuParallelLauncher

# address_by_interface is needed for the HighThroughputExecutor:
from parsl.addresses import address_by_interface


if __name__ == "__main__":
    run_dir = "/eagle/projects/argonne_tpc/mansisak/memorization/src/localize/"
    env = "/grand/SuperBERT/mansisak/memorization/env/"

    user_opts = {
        "worker_init": f"module use /soft/modulefiles; module load conda; conda activate {env}; cd {run_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "argonne_tpc",
        "queue": "preemptable",  # e.g.: "prod","debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "24:00:00",
        "nodes_per_block": 1,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
        # "cpus_per_node":    32, # Up to 64 with multithreading
        "available_accelerators": 4,  # Each Polaris node has 4 GPUs, setting this ensures one worker per GPU
        # "cores_per_worker": 8, # this will set the number of cpu hardware threads per worker.
    }

    config = Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                heartbeat_period=15,
                heartbeat_threshold=120,
                worker_debug=True,
                max_workers_per_node=4,
                available_accelerators=user_opts[
                    "available_accelerators"
                ],  # if this is set, it will override other settings for max_workers if set
                # cores_per_worker=user_opts["cores_per_worker"],
                address=address_by_interface("bond0"),
                cpu_affinity="block-reverse",
                prefetch_capacity=0,
                provider=PBSProProvider(
                    launcher=MpiExecLauncher(
                        bind_cmd="--cpu-bind", overrides="--depth=64 --ppn 1"
                    ),
                    account=user_opts["account"],
                    queue=user_opts["queue"],
                    select_options="ngpus=4",
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
    ):
        # assign duplication folder or not
        dup_folder = "no_dup_noise"
        if dup == "1":
            dup_folder = "dup_noise"
        if backdoor == "1":
            dup_folder = "no_dup_backdoor"

        # add in ckpt dir derivation
        base_dir = f"{data_name}/{num_7}_{num_extra_data}_{num_extra_data}_{num_extra_data}_{num_extra_data}_{length}_{max_ctx}_{seed}_{batch_size}_{lr}"
        if data_name == "wiki_fast":
            base_dir = f"{data_name}/{length}_{max_ctx}_{seed}_{batch_size}_{lr}"

        base_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/{dup_folder}/{base_dir}/"
        if data_name == "mult" and backdoor == "1":
            trained_epochs = 5
            placeholder_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/5_mult_data_distributions_bd_testing_150/four_layer/"
        if data_name == "mult" and backdoor == "0":
            trained_epochs = 2000
            placeholder_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/5_mult_data_distributions_testing_150/four_layer/"
        if data_name == "wiki_fast" and backdoor == "0":
            trained_epochs = 30
            placeholder_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/lm_test/wiki_4_noise_dup/"
        if data_name == "wiki_fast" and backdoor == "1":
            trained_epochs = 60
            placeholder_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/lm_test/wiki_4_backdoor_dup/"

        if n_layers == "1":
            layer_dir = "one_layer"
        if n_layers == "2":
            layer_dir = "two_layer"
        if n_layers == "4":
            layer_dir = "four_layer"
        if n_layers == "8":
            layer_dir = "eight_layer"
        if n_layers == "16":
            layer_dir = "sixteen_layer"

        # TODO (MS): fix model path!
        ckpt_dir = f"{base_path}{layer_dir}/"
        model_name = f"{n_layers}_layer_{trained_epochs}_epoch.pth"
        model_path = f"{placeholder_path}{model_name}"
        # model_path = f"{ckpt_dir}{model_name}"
        print(model_path)

        exec_str = f"python localize_hp_sweep.py --model_path {model_path} --n_layers {n_layers} --data_name {data_name} --num_extra {num_extra_data} --seed {seed} --duplicate {dup} --backdoor {backdoor}"

        return f" env | grep CUDA; {exec_str};"

    param_list = []

    for layer in [4]:
        for lr in [1e-3]:
            for data_name in ["wiki_fast", "mult"]:
                # for data_name in ["mult", "increment", "wiki_fast"]:
                for batch_size in [32]:
                    for extra_data_size in [20000]:
                        for dup in [0, 1]:
                            for backdoor in [0, 1]:
                                for seed in [
                                    0,
                                ]:

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

                                    args_dict = {
                                        "n_layers": f"{layer}",
                                        "batch_size": f"{batch_size}",
                                        "lr": f"{lr}",
                                        "data_name": f"{data_name}",
                                        "num_7": f"20000",
                                        "num_extra_data": f"{extra_data_size}",
                                        "epochs": f"1",
                                        "seed": f"{seed}",
                                        "length": f"20",
                                        "max_ctx": f"150",
                                        "dup": f"{dup}",
                                        "backdoor": f"{backdoor}",
                                    }
                                    print(args_dict)
                                    param_list.append(args_dict)

    print("Number of total experiments: ", len(param_list))
    futures = [localize_memorization(**args) for args in param_list]

    for future in futures:
        print(f"Waiting for {future}")
        print(f"Got result {future.result()}")

        # with open(future.stdout, "r") as f:
        #    print(f.read())