# Project: multitask reinforcement learning in MetaWorld using dendritic networks.


Experiments with multitask reinforcement learning, using dendritic networks, tested in MetaWorld.

For more details please contact lsouza at numenta dot com.

### Installation

Important: MetaWorld requires Mujoco. For information about installing and acquiring a license, please refer to the [Mujoco github repo](https://github.com/openai/mujoco-py)

Once Mujoco is installed, create a new environment from the yaml file:
`conda env create -f environment.yml`
`conda activate multitask_dendrites`

For proper logging, make sure to set your `WANDB_API_KEY=<my_api_key>` and `WANDB_DIR=<path_to_log_directory>` environment variables. The path set in `WANDB_DIR` needs write permission for the user running the script, so make sure to set the correct permissions (for linux, use the `chmod +rwx <path_to_log_directory>` command.) See [wandbdoc](https://docs.wandb.ai/guides/track/advanced/environment-variables) for more details. If you don't have an account, create (a free) one to get an API key.

To save models, set the `CHECKPOINT_DIR=<path_to_checkpointing_directory>`. As before, make sure to set write permission for the user running the script.

### Execution

To run an experiment, first define a new experiment in a python module under the folder experiments. Please follow the example of other configs already created.

To run, on projects/multitask call `python run.py -e <experiment_name>`

### Results

For results shown in the paper Avoiding Catastrophe: Active Dendrites Enable Multi-task Learning in Dynamic Environments, refer to the [public wandb workspace](https://wandb.ai/nupic-research/multitask_journal)