import hydra
from omegaconf import DictConfig, OmegaConf

from dataset import prepare_dateset
from client import generate_client_fn

@hydra.main(config_path="conf", config_name="base", version_base=None)
def main(cfg: DictConfig):

    ## 1. Parse config & get experiment output dir
    print(OmegaConf.to_yaml(cfg))

    ## 2. Prepare your dataset
    trainloaders, validationloaders, testloader = prepare_dateset(cfg.num_clients, cfg.batch_size)

    ## 3. Define your clients
    client_fn=generate_client_fn(trainloaders, validationloaders, cfg.num_classes)

    ## 4. Define your strategy

    ## 5. Start simulation

    ## 6. Save your results



if __name__ == "__main__":

    main()