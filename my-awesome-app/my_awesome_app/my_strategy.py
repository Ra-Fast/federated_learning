from flwr.common import FitRes, Parameters, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg,FedAvgM,QFedAvg
import torch
import json
import wandb
from datetime import datetime

from .task import Net, set_weights



class CustomFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.results_to_save = {}

        # Log those same metrics to Weights & Biases
        name=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        strategy_name = self.__class__.__name__
        #strategy_name = "QFedAvg_lr_0_5_q_param_1"
        name = f"{strategy_name}_{name}"
        wandb.init(project="flower-simulation-tutorial", name=f"custom-strategy-{name}")

    def aggregate_fit(self, 
                      server_round: int, 
                      results: list[tuple[ClientProxy,FitRes]], 
                      failures:list[tuple[ClientProxy,FitRes] | BaseException]
                      ) -> tuple[Parameters | None, dict[str, bool | bytes | float | int | str]]:
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(server_round, results, failures)
        
        # Convert parameters to ndarrays
        ndarrays = parameters_to_ndarrays(parameters_aggregated)

        # Instantiate model and set weights
        model = Net()
        set_weights(model, ndarrays)

        # Save global model in the standart PyTorch format
        # torch.save(model.state_dict(), f"model_round_{server_round}.pth")

        return parameters_aggregated, metrics_aggregated
    
    def evaluate(self, 
                 server_round: int, 
                 parameters: Parameters
                 ) -> tuple[float, dict[str, bool | bytes | float | int | str]] | None:
        loss, metrics = super().evaluate(server_round, parameters)

        my_results = {"loss": loss, **metrics}

        self.results_to_save[server_round] = my_results

        # Save evaluation results in a JSON file
        with open("result.json", "w") as json_file:
            json.dump(self.results_to_save, json_file,indent=4)

        # Log to W&B
        wandb.log(my_results,step=server_round)
        return loss, metrics
    