from src.data import (
    load_dataset, 
    stratified_folds, 
    stratified_validation_split
)


def class_counts(dataset, indices):
    counts = [
        0 for _ in range(dataset.num_classes)
    ]

    for index in indices:
        class_id = dataset[index].y.item()
        counts[class_id] += 1

    return counts


def main():
    dataset_names = [
        "MUTAG",
        "PROTEINS",
        "NCI1",
    ]

    for dataset_name in dataset_names:
        dataset = load_dataset(dataset_name)
        folds = stratified_folds(dataset, 5, 0)

        all_indices = set(range(len(dataset)))
        all_test_indices = []

        print()
        print("Dataset:", dataset_name)
        print("Graphs:", len(dataset))

        for fold_id, test_indices in enumerate(folds):
            test_set = set(test_indices)

            remainder_indices = [
                index
                for index in range(len(dataset))
                if index not in test_set
            ]

            fit_indices, val_indices = stratified_validation_split(
                dataset,
                remainder_indices,
                1000 + fold_id,
            )

            fit_set = set(fit_indices)
            val_set = set(val_indices)

            disjoint = (
                fit_set.isdisjoint(val_set)
                and fit_set.isdisjoint(test_set)
                and val_set.isdisjoint(test_set)
            )

            complete = (
                fit_set | val_set | test_set
                == all_indices
            )

            all_test_indices += test_indices

            print()
            print("Fold:", fold_id)
            print(
                "Fit:",
                len(fit_indices),
                class_counts(dataset, fit_indices),
            )
            print(
                "Validation:",
                len(val_indices),
                class_counts(dataset, val_indices),
            )
            print(
                "Test:",
                len(test_indices),
                class_counts(dataset, test_indices),
            )
            print("Disjoint:", disjoint)
            print("Complete:", complete)

        outer_partition = (
            len(all_test_indices) == len(dataset)
            and len(set(all_test_indices)) == len(dataset)
        )

        print()
        print("Outer test partition:", outer_partition)


if __name__ == "__main__":
    main()