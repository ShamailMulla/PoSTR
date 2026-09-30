import os
import pickle
import json
import pathlib
from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from .utils import create_directories

if TYPE_CHECKING:
    from ..agent.btrl import BTRL


class DataManager:
    def __init__(self, config: dict):
        if config['load']:
            self.logdir = "./logdir/{}".format(config["experiment"]["env"].replace('/', '-')
                                               ) + "/" + config["algorithm"]["name"] + "-{}".format(
                                            config["experiment"]["name"]) + "/{}/".format(config["experiment"]["seed"])
        else:
            self.logdir = create_directories(
                config["experiment"]["env"],
                config["algorithm"]["name"],
                config["experiment"]["name"],
            )
            with open(self.logdir + "hyper_parameters.txt", "w") as f:
                json.dump(config, f, indent=2)
        self.writer = SummaryWriter(log_dir=self.logdir)

    def update(self, log: dict, timestep: int):
        for key, value in log["scalars"].items():
            if np.isnan(value):
                continue
            self.writer.add_scalar(key, value, timestep)
        with (pathlib.Path(self.logdir) / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"Timestep": timestep, **dict(log["scalars"])}) + "\n")

    def save(self, agent: "BTRL", timestep: int):
        path = self.logdir + "checkpoints/" + str(timestep) + "/"
        # print('path:',path)
        os.mkdir(path)
        print('saving transition.pt')
        torch.save(agent.model.transition_network.state_dict(), path + "transition.pt")
        torch.save(agent.model.terminal_network.state_dict(), path + "terminal.pt")
        torch.save(agent.value_network.state_dict(), path + "value.pt")
        with open(path + "replay.pt", "wb") as fn:
            pickle.dump(agent.dataset.episodes, fn)


