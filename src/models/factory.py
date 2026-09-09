from src.models.gcn import GCN
from src.models.graphsage import GraphSAGE
from src.models.gin import GIN
from src.models.gat import GAT


def build_model(settings, num_features, num_classes):
    if settings["model"] == "GCN":
        return GCN(
            num_features,
            settings["hidden_dim"],
            num_classes,
        )

    elif settings["model"] == "GraphSAGE":
        return GraphSAGE(
            num_features,
            settings["hidden_dim"],
            num_classes,
        )

    elif settings["model"] == "GIN":
        return GIN(
            num_features,
            settings["hidden_dim"],
            num_classes,
        )

    elif settings["model"] == "GAT":
        return GAT(
            num_features,
            settings["hidden_dim"],
            num_classes,
            heads=settings["heads"],
        )

    raise ValueError(f"Unknown model: {settings['model']}")