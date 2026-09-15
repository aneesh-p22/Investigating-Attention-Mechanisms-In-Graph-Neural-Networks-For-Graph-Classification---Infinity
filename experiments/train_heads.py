from experiments.ablation import head_variants, run_variant


def main():
    print("RQ1 fixed-width GAT head-count fits:")

    for variant in head_variants:
        print()
        print("Variant:", variant["name"])

        run_variant(variant)


if __name__ == "__main__":
    main()