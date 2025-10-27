import os
import hydra


def run_train(cfg):
    pass

@hydra.main(config_path='config',config_name='config.yaml')
def main(cfg):
    run_train(cfg)

if __name__ == '__main__':
    main()




























