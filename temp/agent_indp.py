import os 
import sys
sys.append(os.getcwd())

import numpy as np
import torch
from data.helpers import dodict
import pdb

class AgentNetwork(nn.Module):
    def __init__(self, input_dims, output_dims, agent_network):
        """__init__.

        Args:
            input_dims:
            output_dims:
            agent_network:
        """
        super().__init__()
        self.input_dims = input_dims
        self.ouptut_dims = output_dims
        self.agent_network = agent_network
        # Initialize model architecture here.
        self.net = nn.ModuleList()
        idim = self.input_dims[0]
        if len(self.input_dims) != 1:
            # Adding Convolutional Layers 
            clayers = self.agent_network.clayers
            cl_dims = self.agent_network.cl_dims
            for c in range(clayers):
                self.net.append(
                        nn.Conv2D())
                idim = cl_dims[c]

            self.net.append(nn.Flatten())
            idim = cl_dims[-1] * \
                    ((self.input_dims[-1]-int(clayers))**2)
        nlayers = self.agent_network.nlayers
        nl_dims = self.agent_network.nl_dims
        for l in range(nlayers):
            self.net.append(
                    nn.Linear(idim, nl_dims[l]))
            idim = nl_dims[l]
        self.final_layer = nn.Linear(nl_dims[-1], self.ouptut_dims)

    def forward(self, inputs):
        for layer in self.net:
            x = layer(inputs)
            inputs = x 
        logits = self.final_layer(x)
        return F.softmax(logits, dim=1)

class EvalAgent():
    def __init__(self, _id, input_dims, output_dims, model_path):
        """__init__.
        A Agent class for Testing Independent Agent Policies. Independence here implies that the agents are not 
        networked by any means (communication channels, etc.) and can act in the environment in a decentralized manner.
        The model will be loaded from the "model" argument and a dimension check will be performed. If unsucessfull, the 
        program will terminate with an error message.
        Args:
            _id: Agent _id as seen in the game.
            input_dims: The current inputs dims of the game.
            output_dims: The current output_dims of the game.
            model_path: The Path to the model (torch.model) to be loaded.
        """
        self._id = _id
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.model_path = model_path
        # Loading the model here.
        model = torch.load(f"trained_policies/"+self.model)
        breakpoint()
        self.network_dims = dodict(model['agent_network'])
        # Check if the agent network dims match with the current envrionment input and output shapes.
        self.network = NetworkAgent(self.input_dims, self.output_dims, self.agent_network)
        self.network.load_state_dict(model['model_state_dict'])
        self.network.eval()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.network = self.network.to(self.device)
        # Print that the model has been loaded successfully

    
    def get_action(self, observation):
        """get_action.
        Certain networks might output two values (combined probabilites and values), 
        this must be changed here in this loop.
        Args:
            observation:
        """
        observation = torch.as_tensor(observation, dtpye=torch.float32, device = self.device)
        probs, _ = self.network(obesrvation.unsqueeze(0))
        action_dist = dist.Categorical(probs)
        action = aciton_dist.sample()
        log_probs = action_dist.log_prob(action)
        return action.item()
