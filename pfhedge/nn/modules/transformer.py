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

from transformers.models.time_series_transformer.modeling_time_series_transformer import (
    TimeSeriesTransformerDecoder,
    _make_causal_mask,
)

from transformers.models.time_series_transformer.configuration_time_series_transformer import (
    TimeSeriesTransformerConfig,
)

class Transformer(Module):
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

        config = TimeSeriesTransformerConfig(
            context_length=100,
            prediction_length=50,
            d_model = n_units,
            decoder_layers=n_layers,
            feature_size = in_features,
        )

        self.transformer = TimeSeriesTransformerDecoder(config)
        self.projection = Sequential(
            Linear(n_units,out_features),
            out_activation,
        )
    
    def forward(
        self,
        model_input, # [N,TimeStep,Features]
    ):

        model_output = self.transformer(
            inputs_embeds = model_input,
        ).last_hidden_state
        model_output = self.projection(model_output)
        return model_output