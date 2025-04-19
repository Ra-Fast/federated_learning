"""my-awesome-app: A Flower / PyTorch app."""
from typing import List, Tuple
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from my_awesome_app.task import Net, get_weights, set_weights, test, get_transforms
from datasets import load_dataset
from torch.utils.data import DataLoader
from my_awesome_app.my_strategy import CustomFedAvg

def get_evaluate_fn(testloader, device):
    """Callback for evaluation function for server-side evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        """Evaluate the model using the provided test data."""
        # Set model parameters
        net=Net()
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, accuracy = test(net, testloader, device)
        # Return loss and accuracy
        return loss, {"cen_accuracy": accuracy}
    return evaluate

def weights_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """ A function that aggregate metrics"""
    accuracies= [num_examples * m["accuracy"] for num_examples, m in metrics]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    accuracy = sum(accuracies) / total_examples if total_examples > 0 else 0.0
    return {"accuracy": accuracy}

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    for _, m in metrics:
        print(m)

    return {}

def on_fit_config(server_round: int) -> Metrics:
    """Adjust learning rate based on server round."""
    lr=0.01
    if server_round > 2:
        lr = 0.005
    return{"lr": lr}  

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)

    # Load global test set
    testset=load_dataset("uoft-cs/cifar10")["test"]
    testloader=DataLoader(testset.with_transform(get_transforms()), batch_size=32)
    

    # Define strategy
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weights_average,
        fit_metrics_aggregation_fn=handle_fit_metrics,
        #on_fit_config_fn=on_fit_config,
        evaluate_fn=get_evaluate_fn(testloader, device='cpu'),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
