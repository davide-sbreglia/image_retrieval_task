import wandb

if __name__ == "__main__":
    wandb.init(
        project="ml-competizione-retrieval",
        entity="davide-sbreglia-university-of-trento",  # <-- usa il tuo username wandb
        config={
            "model": "resnet50-baseline",
            "batch_size": 64,
            "lr": 1e-3
        }
    )
    print("✅ W&B init done.")
