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
    run_dir = "/eagle/projects/argonne_tpc/mansisak/memorization/src/"
    env = "/grand/SuperBERT/mansisak/memorization/env/"

    user_opts = {
        "worker_init": f"module use /soft/modulefiles; module load conda; conda activate {env}; cd {run_dir}",  # load the environment where parsl is installed
        "scheduler_options": "#PBS -l filesystems=home:eagle:grand",  # specify any PBS options here, like filesystems
        "account": "SuperBERT",
        "queue": "debug",  # e.g.: "debug, "preemptable" (see https://docs.alcf.anl.gov/polaris/running-jobs/)
        "walltime": "01:00:00",
        "nodes_per_block": 2,  # think of a block as one job on polaris, so to run on the main queues, set this >= 10
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
    def generate_completions(
        # model_name="EleutherAI/gpt-neo-125M",
        # path_to_prompts="data/prompts/prompts_100.npy",
        # generations_save_path="/grand/projects/SuperBERT/aswathy/projects/memorization/data/model_generations2/gpt-neo-125M/125M-0.0_prompt_50_of_100.npy",
        # mem_prompt_save_path="data/memorized_prompts/gpt-neo-125M/125M-0.0_mem_50_of_100.npy",
        # stdout="echo-hello.stdout",
        # stderr="echo-hello.stderr",
        n_layers=1,
    ):
        # exec_str = f"python -m src.run_model_generations --model_name {model_name} --path_to_prompts {path_to_prompts} --generations_save_path {generations_save_path} --mem_prompt_save_path {mem_prompt_save_path}"
        exec_str = f"python memorization_in_toy_models.py --max_ctx 150 --data_name wiki_fast --n_layers {n_layers} --ckpt_dir wiki_fast_{n_layers} --vocab_size 50257"

        return f" env | grep CUDA; {exec_str};"

    """
    models = ["EleutherAI/gpt-neo-125M"]
    model_sizes = [model.split("-")[-1] for model in models]
    param_list = []

    for model, model_size in zip(models, model_sizes):
        for seq_len in range(100, 350, 50):
            for p_len in range(50, seq_len, 50):
                model_name_in_path = model.split("/")[-1]
                args_dict = {
                    "model_name": model,
                    "path_to_prompts": f"data/prompts/prompts_{seq_len}.npy",
                    "generations_save_path": f"data/model_generations2/{model_name_in_path}/{model_size}-0.0_prompt_{p_len}_of_{seq_len}.npy",
                    "mem_prompt_save_path": f"data/memorized_prompts/{model_name_in_path}/{model_size}-0.0_mem_{p_len}_of_{seq_len}.npy",
                    "stdout": f"mem-parsl-{model_name_in_path}/{p_len}-of-{seq_len}.stdout",
                    "stderr": f"mem-parsl-{model_name_in_path}/{p_len}-of-{seq_len}.stderr",
                }
                param_list.append(args_dict)
    """
    param_list = []

    for layer in [1, 2, 4, 8, 16]:
        args_dict = {
            "n_layers": f"{layer}",
        }
        param_list.append(args_dict)

    futures = [generate_completions(**args) for args in param_list]

    for future in futures:
        print(f"Waiting for {future}")
        print(f"Got result {future.result()}")

        with open(future.stdout, "r") as f:
            print(f.read())
