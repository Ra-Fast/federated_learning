
from omegaconf import DictConfig
from model import test
from hydra.utils import instantiate
import torch
from collections import OrderedDict
def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        
        return{'lr': config.lr, 'momentum': config.momentum, 'local_epochs': config.local_epochs}
    
    return fit_config_fn


def get_evaluate_fn(model_cfg: int, testloader):
    def evaluate_fn(server_round: int, parameters, config):

        model=instantiate(model_cfg)
        
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        params_dict=zip(model.state_dict().keys(), parameters)
        state_dict=OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict,strict=True)

        loss, accuracy=test(model, testloader, device)
        return loss, {'accuracy': accuracy}
    return evaluate_fn