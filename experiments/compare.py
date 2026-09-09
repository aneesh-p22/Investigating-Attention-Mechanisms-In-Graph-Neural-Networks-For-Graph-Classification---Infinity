import json


result_paths = [
    "results/development_gcn_mutag_seed0.json",
    "results/development_graphsage_mutag_seed0.json",
    "results/development_gin_mutag_seed0.json",
]

print("Saved development results:")

for result_path in result_paths:
    with open(result_path, encoding="utf-8") as result_file:
        result = json.load(result_file)

    settings = result["settings"]
    selection = result["selection"]
    parameters = result["parameters"]

    print()
    print("Model:", settings["model"])
    print("Dataset:", settings["dataset"])
    print("Training seed:", settings["seed"])
    print("Split seed:", settings["split_seed"])
    print("Selected epoch:", selection["selected_epoch"])

    print(
        f"Validation loss: "
        f"{selection['validation_loss']:.4f}"
    )

    print(
        f"Validation accuracy: "
        f"{selection['validation_accuracy']:.4f}"
    )

    print("Total parameters:", parameters["total"])
    print("Trainable parameters:", parameters["trainable"])
    print("Result:", result_path)