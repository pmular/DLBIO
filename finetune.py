import subprocess
import wandb

# This function will be called by wandb for each set of hyperparameters
def train():
    # Retrieve hyperparameters from wandb
    run = wandb.init(reinit=True)  # This initializes the run and allows reinitialization
    config = run.config

    # Construct the command to run `run.py` with the given config
    command = [
        "python3", "run.py",
        "exp.name=finetune_sweeps_tm",
        "method=protonet_sot",
        "dataset=tabula_muris",
        f"method.cls.self_ot.reg={config['reg']}",
        f"optimizer_cls.lr={config['lr']}",
        f"backbone.layer_width={config['layer_width']}",
        f"backbone.depth={config['depth']}",
        f"backbone.dropout={config['dropout']}",
    ]

    # Run the command
    subprocess.run(command)
    run.finish()  # This ends the current run



def main():
    # Define the sweep configuration
    sweep_config = {
        'method': 'random',  # or 'random' or 'bayes'
        'metric': {
            'name': 'accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'reg': {
                'values': [0.05, 0.1, 0.2]
            },
            'lr': {
                'values': [0.005, 0.001, 0.0005]
            },
            'layer_width': {
                'values': [32, 64, 128, 256]
            },
            'depth': {
                'values': [1, 2]
            },
            'dropout': {
                'values': [0.0, 0.1, 0.3]
            },
        }
    }

    wandb.login(key="515249130d470758f03aca4f951e814d24ea9ed2")
    sweep_id = wandb.sweep(sweep_config, project='dlbio', entity='pma3')

    # Run the sweep
    wandb.agent(sweep_id, function=train)


if __name__ == "__main__":
    main()
