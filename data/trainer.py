import os
import sys
import numpy as np
import logging
import wandb
from datetime import datetime
from data.helpers import dodict
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, config):
        self.config = config
        self.logger = self.get_logger()

    def get_logger(self):
        if self.config.wandb:
            wandb.init(project=self.config.project_name,
                    notes=self.config.notes,
                    mode=self.config.wandb_mode,
                    config=self.config
                    entity=self.config.entity)
            wandb.run.name = self.config.wandb_run_name
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(message)s')
        logger.setLevel(self.config.log_level)
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"{__name__}:{self.config.msg}")
        logger.info(datetime.now().strftime("%d/%m %H:%M"))
        print(f"{self.config.msg}")
        return logger

    def shut_logger(self):
        if self.config.wandb:
            wandb.finish()
        logging.shutdown()
    
    def update_logs(self, epoch, **kwargs):
        if self.config.wandb:
            wandb.log(dict(epochs=epoch, 
                **kwargs))
        self.logger.info(f"{epoch} | {kwargs}")
        pass
        
    def log(self, _str):
        self.logger.info(_str)
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def initialize_agents(self):
        pass
    
    @abstractmethod
    def run_episodes(self):
        pass
    
    @abstractmethod
    def run_training(self):
        pass

