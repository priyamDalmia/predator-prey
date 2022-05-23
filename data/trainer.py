import os
import sys
import numpy as np
import pandas as pd
import logging
import wandb
from datetime import datetime
from data.helpers import dodict
from abc import ABC, abstractmethod

class Trainer(ABC):
    def __init__(self, env, config):
        self.config = config
        self.env = env
        self.logger = self.get_logger()
        self.agent_ids = env.agent_ids
        self.pred_ids = self.agent_ids[:self.config.npred]
        self.prey_ids = self.agent_ids[self.config.npred:]
        self.train_ids = self.pred_ids \
                if self.config.train_type=="predator" else self.prey_ids
        self.checkpoint_state = None

        self.best_ = 1000 if self.config.train_type=="predator" else 1
        self.steps_avg = 0 
        self.rewards_avg = 0
        self.loss_avg = 0
    
    def train(self):
        pass

    def initialize_agents(self):
        pass
    
    def run_episodes(self):
        pass
    
    def run_training(self):
        pass
    
    def action_to_string(self):
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
        if epoch%400 == 0:
            self.logger.info(f"{epoch} | {kwargs}")
        pass
        
    def make_log(self, epoch, steps_hist, rewards_hist, loss_hist, **kwargs):
        self.steps_avg = np.mean(steps_hist[-49:])
        self.rewards_avg = pd.DataFrame(rewards_hist[-49:], columns=self.agent_ids)\
                .mean(0).round(2).to_dict()
        self.loss_avg = pd.DataFrame(loss_hist[-49:], columns=self.train_ids)\
                .mean(0).round(3).to_dict()
        info = dict(
                steps = self.steps_avg,
                rewards = self.rewards_avg,
                loss = self.loss_avg)
        self.update_logs(epoch, info=info)
        if self.config.print_console:
            print(\
                    f"Epochs:{epoch:4} | Steps:{self.steps_avg:4.2f} | Rewards:{self.rewards_avg}")
        pass

    def log_write(self, msg):
        if self.config.wandb:
            wandb.log({"logs": msg})
        self.logger.info(msg)
    
    def log_model(self, model, **kwargs):
        if self.config.wandb:
            wandb.watch(model, log="all")

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
