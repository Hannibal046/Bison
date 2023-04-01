# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch

# %%
seaborn.set_style("whitegrid")

FONTSIZE = 18
matplotlib.rcParams["figure.figsize"] = (10, 5)
matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["figure.titlesize"] = FONTSIZE
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["legend.fontsize"] = FONTSIZE
matplotlib.rcParams["xtick.labelsize"] = FONTSIZE
matplotlib.rcParams["ytick.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.labelsize"] = FONTSIZE
matplotlib.rcParams["axes.titlesize"] = FONTSIZE
matplotlib.rcParams["savefig.bbox"] = "tight"
matplotlib.rcParams["savefig.pad_inches"] = 0.1
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["axes.linewidth"] = 1.6

# %%
torch.manual_seed(42)

if not torch.cuda.is_available():
    raise RuntimeWarning(
        "CUDA is not available. "
        "If you're using Google Colab, you can enable GPUs as: "
        "https://colab.research.google.com/notebooks/gpu.ipynb"
    )

DEVICE = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print("Default device:", DEVICE)

# %%
# In each epoch, N_PATHS brownian motion time-series are generated.
N_PATHS = 50000
# How many times a model is updated in the experiment.
N_EPOCHS = 200

# %%
def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

# %% [markdown]
# ## How to Use

# %% [markdown]
# ### Prepare Instruments

# %% [markdown]
# We consider a `BrownianStock`, which is a stock following the geometric Brownian motion, and a `EuropeanOption` which is contingent on it.
# 
# We assume that the stock has a transaction cost given by `cost`.

# %%
from pfhedge.instruments import BrownianStock, EuropeanOption

stock = BrownianStock(cost=1e-3)
derivative = EuropeanOption(stock).to(DEVICE)

# %%
derivative

# %% [markdown]
# ### Create Your Hedger

# %% [markdown]
# We here use a multi-layer perceptron as our model.

# %%
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron

model = MultiLayerPerceptron(in_features=4)
hedger = Hedger(
    model, inputs=["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
).to(DEVICE)

# Hedging a derivative with multiple instruments.
# from pfhedge.nn import Hedger
# from pfhedge.nn import MultiLayerPerceptron

# from pfhedge.instruments import HestonStock
# from pfhedge.instruments import EuropeanOption
# from pfhedge.instruments import VarianceSwap
# from pfhedge.nn import BlackScholes

# _ = torch.manual_seed(42)
# stock = HestonStock(cost=1e-4)
# option = EuropeanOption(stock)
# varswap = VarianceSwap(stock)
# pricer = lambda varswap: varswap.ul().variance - varswap.strike
# varswap.list(pricer, cost=1e-4)
# model = MultiLayerPerceptron(3, 2)
# model.to(DEVICE)
# hedger = Hedger(model,
#                 inputs=["moneyness", "time_to_maturity", "volatility"])
# hedger.price(option, hedge=[stock, varswap], n_paths=2)

# %% [markdown]
# The `hedger` is also a [`Module`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module).

# %%
hedger

# %%
history = hedger.fit(derivative, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

# %%
plt.plot(history)
plt.xlabel("Number of epochs")
plt.ylabel("Loss (entropic risk measure)")
plt.title("Loss histories for a European option")
plt.show()

# %%
pnl = hedger.compute_pnl(derivative, n_paths=50000)

plt.figure()
plt.hist(to_numpy(pnl), bins=100)
plt.title("Profit-loss histograms of 50000 price paths for a European option")
plt.xlabel("Profit-loss")
plt.ylabel("Number of events")
plt.show()

# %%
price = hedger.price(derivative)
price

# %% [markdown]
# ## More Examples

# %% [markdown]
# ### Black-Scholes' Delta-Hedging Strategy

# %%
from pfhedge.nn import Hedger
from pfhedge.nn import BlackScholes

model = BlackScholes(derivative)
hedger = Hedger(model, inputs=model.inputs()).to(DEVICE)

# %%
hedger

# %%
price = hedger.price(derivative)
price

# %% [markdown]
# ### Whalley-Wilmott's Asymptotically Optimal Strategy for Small Costs

# %%
from pfhedge.nn import Hedger
from pfhedge.nn import WhalleyWilmott

model = WhalleyWilmott(derivative)
hedger = Hedger(model, inputs=model.inputs()).to(DEVICE)

# %%
price = hedger.price(derivative)
price

# %% [markdown]
# ### Your Own Module

# %%
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Module

from pfhedge.nn import BlackScholes, Clamp, MultiLayerPerceptron


class NoTransactionBandNet(Module):
    def __init__(self, derivative):
        super().__init__()

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        return self.clamp(prev_hedge, min=min, max=max)

# %%
model = NoTransactionBandNet(derivative)
hedger = Hedger(model, inputs=model.inputs()).to(DEVICE)

# %%
history = hedger.fit(derivative, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

# %%
plt.plot(history)
plt.xlabel("Number of epochs")
plt.ylabel("Loss (entropic risk measure)")
plt.title("Loss histories for a European option")
plt.show()

# %%
pnl = hedger.compute_pnl(derivative, n_paths=50000)

plt.figure()
plt.hist(to_numpy(pnl), bins=100)
plt.title("Profit-loss histograms of 50000 price paths for a European option")
plt.xlabel("Profit-loss")
plt.ylabel("Number of events")
plt.show()

# %%
price = hedger.price(derivative)
price

# %% [markdown]
# ### Use Expected Shortfall as a Loss function

# %%
from pfhedge.nn import ExpectedShortfall

# %%
# Expected shortfall with the quantile level of 10%
expected_shortfall = ExpectedShortfall(0.1)

model = NoTransactionBandNet(derivative)
hedger = Hedger(model, inputs=model.inputs(), criterion=expected_shortfall).to(DEVICE)

# %%
history = hedger.fit(derivative, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

# %%
plt.plot(history)
plt.xlabel("Number of epochs")
plt.ylabel("Loss (entropic risk measure)")
plt.title("Loss histories for a European option")
plt.show()

# %%
pnl = hedger.compute_pnl(derivative, n_paths=50000)

plt.figure()
plt.hist(to_numpy(pnl), bins=100)
plt.title("Profit-loss histograms of 50000 price paths for a European option")
plt.xlabel("Profit-loss")
plt.ylabel("Number of events")
plt.show()

# %%
price = hedger.price(derivative)
price

# %%


# %%
from pfhedge.instruments import HestonStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import VarianceSwap
from pfhedge.nn import BlackScholes

_ = torch.manual_seed(42)
stock = HestonStock(cost=1e-4)
option = EuropeanOption(stock)
varswap = VarianceSwap(stock)
pricer = lambda varswap: varswap.ul().variance - varswap.strike
varswap.list(pricer, cost=1e-4)
hedger = Hedger(
        MultiLayerPerceptron(3, 2),
        inputs=["moneyness", "time_to_maturity", "volatility"])
hedger.price(option, hedge=[stock, varswap], n_paths=2)

# %%



