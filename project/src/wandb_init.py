import wandb

if __name__ == "__main__":
    wandb.init(
        project="ml-competizione-retrieval",
        entity="davide-sbreglia-university-of-trento",  # <-- usa il tuo username wandb
        name="submission-selection",
        config={
            "models": ["resnet50", "efficientnet_b0", "vit_b_16", "resnet+vit", "eff+vit"],
            "evaluation_metric": "topk_accuracy",
            "k": 3,
            "dataset": "./test/query vs gallery"
        }
    )
    print("✅ W&B init done for submission benchmarking.")

