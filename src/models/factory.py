from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gin import GIN


def build_model(settings, num_features, num_classes):
    if settings["model"] == "GCN":
        return GCN(
            num_features,
            settings["baseline_width"],
            num_classes,
        )

    elif settings["model"] == "GraphSAGE":
        return GraphSAGE(
            num_features,
            settings["baseline_width"],
            num_classes,
        )

    elif settings["model"] == "GIN":
        return GIN(
            num_features,
            settings["baseline_width"],
            num_classes,
        )

    raise ValueError(f"Unknown model: {settings['model']}")