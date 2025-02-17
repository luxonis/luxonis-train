from luxonis_ml.data import LuxonisDataset

from luxonis_train.core import LuxonisModel


def test_embeddings_model(embedding_dataset: LuxonisDataset):
    model = LuxonisModel(
        cfg="configs/embeddings_model.yaml",
        opts={
            "loader.params.dataset_name": embedding_dataset.dataset_name,
            "trainer.epochs": 1,
            "trainer.validation_interval": 1,
        },
    )
    model.train()
