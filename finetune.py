import subprocess
import wandb

# This function will be called by wandb for each set of hyperparameters
def train():
    # Retrieve hyperparameters from wandb
    run = wandb.init(reinit=True)  # This initializes the run and allows reinitialization
    config = run.config

    # Convert the list of layer dimensions to a Hydra-friendly format
    layer_dim_str = ','.join(map(str, config['layer_dim']))

    # Construct the command to run `run.py` with the given config
    command = [
        "python3", "run.py",
        "exp.name=test_finetune",
        "method=protonet_sot",
        "dataset=swissprot",
        f"method.cls.self_ot.reg={config['reg']}",
        f"optimizer_cls.lr={config['lr']}",
        f"backbone.layer_dim=[{layer_dim_str}]",
        f"backbone.dropout={config['dropout']}",
    ]

    # Run the command
    subprocess.run(command)
    run.finish()  # This ends the current run



def main():
    # Define the sweep configuration
    sweep_config = {
        'method': 'bayes',  # or 'random' or 'bayes'
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'reg': {
                'values': [0.05, 0.1, 0.15]
            },
            'lr': {
                'values': [0.005, 0.001, 0.0005]
            },
            'layer_dim': {
                'values': [[64,64], [256], [128], [128, 128]]
            },
            'dropout': {
                'values': [0.0, 0.1, 0.3]
            },
        }
    }

    wandb.login()
    sweep_id = wandb.sweep(sweep_config, project='DL for Bio', entity='kenjitetard0')

    # Run the sweep
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()
