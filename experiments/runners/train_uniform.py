from experiments.ablation import run_variant, uniform_variant


def train_uniform():
    print("RQ3 uniform-attention GAT fits:")
    print()
    print("Variant:", uniform_variant["name"])

    run_variant(uniform_variant)


if __name__ == "__main__":
    train_uniform()