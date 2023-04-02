from copy import deepcopy
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Linear
from torch.nn import Module
from torch.nn import ReLU,Tanh
from torch.nn import RNN
from torch.nn import Sequential

class ElmanRNN(Module):

    def __init__(
        self,
        in_features=None,
        out_features: int = 1,
        n_layers: int = 4,
        n_units: Union[int, Sequence[int]] = 32,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ) -> None:
        
        super().__init__()

        if isinstance(activation,ReLU):
            activation='relu'
        elif isinstance(activation,Tanh):
            activation='tanh'

        self.rnn = RNN(
            input_size = in_features,
            hidden_size = n_units,
            num_layers = n_layers,
            nonlinearity = activation,
            batch_first=True,   
            dropout=0.1,
        )

        self.out_projection = Sequential(
            Linear(n_units,out_features),
            out_activation,
        )
        
    
    def forward(
            self,
            model_input # [N,TimeStep,Features]
        ):
        
        output,_ = self.rnn(model_input) # [N,T,HiddenSize]
        output = self.out_projection(output) # [N,T,OutFeatures]
        return output



