import subprocess
import wandb

def run_experiment(config):
    # Construct the command to run `run.py` with the given config
    command = [
        "python3", "run.py",
        f"exp.name= test_finetune",
        f"method=protonet_sot",
        f"dataset=swissprot",
        f"self_ot.reg={config['reg']}",
        f"self_ot.diag_weight={config['diag_weight']}",
        f"self_ot.tol={config['tol']}",
        f"self_ot.tau={config['tau']}",
        f"optimizer_cls.lr={config['lr']}",
        f"self_ot.tol={config['tol']}"

    ]

    # Run the command
    subprocess.run(command)

def main():
    # Define a set of hyperparameters to test
    hyperparameters = [
        {'reg': 0.05, 'diag_weight': 50, 'tol': 1e-5, 'tau': 10, 'lr':0.001},
        {'reg': 0.1, 'diag_weight': 100, 'tol': 1e-6, 'tau': 100, 'lr':0.001},
        # Add more configurations as needed
    ]

    # Initialize a wandb project
    wandb.init(project='your_project_name', entity='your_wandb_entity')

    # Iterate over the hyperparameters and run experiments
    for config in hyperparameters:
        # Log the current configuration to wandb
        wandb.config.update(config)

        # Run the experiment with the current configuration
        run_experiment(config)

    # Finish the wandb session
    wandb.finish()

if __name__ == "__main__":
    main()
