import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from pfhedge.instruments import BrownianStock, EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron,ElmanRNN,Transformer


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

torch.manual_seed(42)

if not torch.cuda.is_available():
    raise RuntimeWarning(
        "CUDA is not available. "
        "If you're using Google Colab, you can enable GPUs as: "
        "https://colab.research.google.com/notebooks/gpu.ipynb"
    )

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Default device:", DEVICE)

N_PATHS = 50000
N_EPOCHS = 200

def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

stock = BrownianStock(cost=1e-3)
derivative = EuropeanOption(stock).to(DEVICE)

inputs=["log_moneyness", "expiry_time", "volatility"]
inputs.append("prev_hedge")

# model = ElmanRNN(in_features=len(inputs))
# model = MultiLayerPerceptron(in_features=len(inputs))
model = Transformer(in_features=len(inputs))

hedger = Hedger(
    model, 
    inputs=inputs,
    sequence_prediction=True,
).to(DEVICE)

print(model)
history = hedger.fit(derivative, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

pnl = hedger.compute_pnl(derivative, n_paths=N_PATHS)
print('P&L mean: {}, std: {}'.format(round(np.nanmean(to_numpy(pnl)), 4), round(np.nanstd(to_numpy(pnl)), 4)))
