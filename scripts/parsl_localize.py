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
        "account": "superbert",
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
                label="localize_toy",
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
        ckpt_epoch=100,
    ):
        def check_if_hp_run_fin(csv):
            import pandas as pd

            total_exp = 0
            total_finished = 0

            df = pd.read_csv(csv)
            mask = df.localization_method == "base_stats"

            # NOTE(MS): this will only work for no dup categories (for lang will need to make it more fancy)
            # here we check if there is any mem at all -- if not, then we continue
            if df.iloc[0].data_name in ["mult", "increment"]:
                if (df[mask].perc_mem_0 == 0).any():
                    print(f"total finished: {total_finished}/{total_exp}")
                    return 1
            else:
                if "backdoor" in df.model_path[0]:
                    if (df[mask].perc_mem_0 == 0).any():
                        print(f"total finished: {total_finished}/{total_exp}")
                        return 1
                else:  # if we have duplication, check if the last category has memorization
                    if (df[mask].perc_mem_3 == 0).any():
                        print(f"total finished: {total_finished}/{total_exp}")
                        return 1

            for loc_method in [
                "zero",
                "act",
                "hc",
                "slim",
                "durable",
                "durable_agg",
                "random",
                "random_greedy",
                "greedy",
                "ig",
                "obs",
            ]:
                for ratio in [
                    0.00001,
                    0.0001,
                    0.001,
                    0.01,
                    0.05,
                    0.1,
                    0.25,
                    0.3,
                    0.5,
                    0.8,
                ]:
                    if "16" in df.model_path[0]:
                        if loc_method in ["ig"]:
                            continue

                    if "wiki" in df.iloc[0].data_name:
                        if loc_method in ["greedy"]:
                            if ratio >= 0.05:
                                continue

                    if loc_method not in ["random", "random_greedy"]:
                        if ratio >= 0.1:
                            continue

                    # this ratio is too small for neuron-level methods
                    if loc_method in ["zero", "hc", "ig", "slim", "act"]:
                        if ratio <= 0.0001:
                            continue

                    if loc_method in ["greedy"]:
                        if ratio > 0.05:
                            continue

                    if loc_method in ["slim", "hc", "random"]:
                        for epochs in [1, 10, 20]:
                            total_exp += 1
                            if (
                                (df["ratio"] == ratio)
                                & (df["localization_method"] == loc_method)
                                & (df["epochs"] == epochs)
                            ).any():
                                # print("finsihed")

                                total_finished += 1
                    if loc_method in ["random_greedy"]:
                        for epochs in [1, 10, 20]:
                            for loss_weight in [0.9, 0.7, 0.5]:
                                total_exp += 1
                                if (
                                    (df["ratio"] == ratio)
                                    & (df["localization_method"] == loc_method)
                                    & (df["epochs"] == epochs)
                                    & (df["loss_weighting"] == loss_weight)
                                ).any():
                                    # print("finsihed")
                                    total_finished += 1
                            # here we check ratio, loc_method, epoch

                    if loc_method in [
                        "zero",
                        "act",
                        "durable",
                        "durable_agg",
                        "obs",
                        "greedy",
                        "ig",
                    ]:
                        total_exp += 1
                        if (
                            (df["ratio"] == ratio)
                            & (df["localization_method"] == loc_method)
                        ).any():
                            # print("finsihed")
                            total_finished += 1

            # print("total experiments: ", total_exp)
            print(f"total finished: {total_finished}/{total_exp}")
            if total_exp == total_finished:
                return 1
            return 0

        # assign duplication folder or not
        if dup == "1":
            dup_folder = "dup"
        if dup == "0":
            dup_folder = "not_dup"
        if backdoor == "1":
            backdoor_folder = "backdoor"
        if backdoor == "0":
            backdoor_folder = "noise"

        # add in ckpt dir derivation
        base_dir = f"{data_name}/{num_7}_{num_extra_data}_{length}_{max_ctx}_{seed}_{batch_size}_{lr}"
        if data_name == "wiki_fast":
            base_dir = f"{data_name}/{length}_{max_ctx}_{seed}_{batch_size}_{lr}"

        # NOTE(MS): (fix dir once we have final trained models)
        base_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/{backdoor_folder}/{dup_folder}/{base_dir}/"
        # base_path = f"/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/old_train_run/{backdoor_folder}/{dup_folder}/{base_dir}/"

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

        ckpt_dir = f"{base_path}{layer_dir}/"
        model_name = f"{n_layers}_layer_{ckpt_epoch}_epoch.pth"
        model_path = f"{ckpt_dir}{model_name}"
        print("This is model path: ", model_path)

        # check if model path exists before starting experiment
        edit_ckpt_dir = f"{base_path}{layer_dir}_edit/"
        results_path = f"{edit_ckpt_dir}localization_results_{ckpt_epoch}.csv"

        import os

        if os.path.isfile(model_path):
            # print("model_exists")
            if os.path.isfile(results_path):
                print("RESULTS EXIST:", results_path)
                if check_if_hp_run_fin(results_path):
                    print("finished all experiments, moving onto next model")
                    exec_str = "echo finished experiment for model"
                    return f" env | grep CUDA; {exec_str};"

        if os.path.isfile(model_path):
            exec_str = f"python localize_hp_sweep.py --model_path {model_path} --n_layers {n_layers} --data_name {data_name} --num_extra {num_extra_data} --seed {seed} --duplicate {dup} --backdoor {backdoor}"
        else:
            print("file doesn't exist")
            exec_str = "echo file does not exist"

        return f" env | grep CUDA; {exec_str};"

    param_list = []

    for seed in [
        0,
        1,
        3,
        # 3,
        # 4,
    ]:
        for lr in [1e-3]:
            for batch_size in [128]:
                for extra_data_size in [3000, 10000, 20000]:
                    for dup in [0, 1]:
                        for backdoor in [1, 0]:
                            for data_name in ["mult", "increment", "wiki_fast"]:
                                for layer in [4, 16, 2, 8]:
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
