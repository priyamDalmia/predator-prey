import os
import sys
import numpy as np
import logging
import wandb
from datetime import datetime
from data.helpers import dodict
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, env, config):
        self.env = env
        self.agent_ids = env.agent_ids
        self.pred_ids = self.agent_ids[:self.config.npred]
        self.prey_ids = self.agent_ids[self.config.nprey:]
        self.config = config
        self.logger = self.get_logger()
    
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

    def get_logger(self):
        if self.config.wandb:
            wandb.init(project=self.config.project_name,
                    notes=self.config.notes,
                    mode=self.config.wandb_mode,
                    config=self.config,
                    entity=self.config.entity)
            wandb.run.name = self.config._name
        logger = logging.getLogger(__name__)
        formatter = logging.Formatter('%(message)s')
        logger.setLevel(self.config.log_level)
        file_handler = logging.FileHandler(self.config.log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"{__name__}:{self.config.notes}")
        logger.info(datetime.now().strftime("%d/%m %H:%M"))
        print(f"{self.config.notes}")
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
        
    def make_log(self, epoch, steps_hist, rewards_hist, loss_hist, **kwargs):
        self.steps_avg = np.mean(steps_hist[-99:])
        self.rewards_avg = pd.DataFrame(rewards_hist[-99:], columns=self.agent_ids)\
                .mean(0).round(0).to_dict()
        self.loss_avg = pd.DataFrame(loss_hist[-99:])\
                .mean(0).round(0).to_dict()
        info = dict(
                steps = self.steps_avg,
                rewards = self.rewards_avg,
                loss = self.loss_avg)
        self.update_logs(epoch, info=info)
        if self.config.print_console:
            print(\
                    f"Epochs:{epoch:4} | Steps:{self.steps_avg:4.2f} | Rewards:{self.rewards_avg}")
        pass
        
    def make_checkpoint(self):
        for _id in self.agent_ids:
            if _id.startswith(self.config.train_type):
                c_name = \
                        f"{_id}-{self.config._name}-{epoch}-{self.steps_avg:.0f}"
                self.agents[_id].save_state(c_name)

    def save_model(self, dir_):
        for _id in self.agent_ids:
            if _id.startswith(config.train_type):
                self.agents[_id].model(dir_)
