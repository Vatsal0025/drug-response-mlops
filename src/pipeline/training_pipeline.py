from src.utils import set_seed
from src.models.gat import GATModel


def run_training():

    set_seed(42)

    print("Training pipeline started...")

    model = GATModel()

    print(model)


if __name__ == "__main__":

    run_training()