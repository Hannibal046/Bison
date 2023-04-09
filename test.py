import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from pfhedge.instruments import BrownianStock, HestonStock, EuropeanOption,GarchStock
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron, ElmanRNN, Transformer, BlackScholes
from pfhedge.nn import ExpectedShortfall

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

torch.manual_seed(527)

if not torch.cuda.is_available():
    raise RuntimeWarning(
        "CUDA is not available. "
        "If you're using Google Colab, you can enable GPUs as: "
        "https://colab.research.google.com/notebooks/gpu.ipynb"
    )

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Default device:", DEVICE)

N_PATHS = 30000
N_EPOCHS = 200
SAMPLE_ROUNDS = 20

def to_numpy(tensor: torch.Tensor) -> np.array:
    return tensor.cpu().detach().numpy()

'''
定义基础
'''

STOCK = GarchStock

stock = {}
stock['nocost'] = STOCK(cost=0)
stock['afcost100'] = STOCK(cost=0.01)
stock['afcost1000'] = STOCK(cost=0.001)

derivative = {}
for key in stock.keys():
    derivative[key] = EuropeanOption(stock[key], maturity=30/250).to(DEVICE)

expected_shortfall = {}
expected_shortfall['50'] = ExpectedShortfall(0.5)
expected_shortfall['99'] = ExpectedShortfall(0.99)

model = {}
hedger = {}
history = {}
pnl = {}

'''benchmark'''
for key in stock.keys():
    name = 'benchmark_{}'.format(key)
    model[name] = BlackScholes(derivative[key])
    hedger[name] = Hedger(model[name], model[name].inputs())

    bm_pnl = hedger[name].compute_pnl(derivative[key], n_paths=N_PATHS)
    with open('./res_data/{}.pkl'.format(name), 'wb') as f:
        pickle.dump(bm_pnl, f)

    price = hedger[name].price(derivative[key], n_paths=N_PATHS * SAMPLE_ROUNDS) * 100
    print('benchmark {} price: '.format(key), price)


inputs=["log_moneyness", "expiry_time", "volatility"]


# '''
# MLP_nopast
# '''
# model['mlp_nopast'] = MultiLayerPerceptron(in_features=len(inputs))
# for v in ['50', '99']:
#     hedger[f'mlp_nopast_es{v}'] = Hedger(
#         model['mlp_nopast'], 
#         inputs=inputs,
#         sequence_prediction=False,
#         criterion=expected_shortfall[v]
#     ).to(DEVICE)

# for hedger_key in ['mlp_nopast_es50', 'mlp_nopast_es99']:
#     for d_key in derivative.keys():
#         name = '{}_{}'.format(hedger_key, d_key)
#         history[name] = hedger[hedger_key].fit(derivative[d_key], n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

#         pnl[name] = hedger[hedger_key].compute_pnl(derivative[d_key], n_paths=N_PATHS)
#         with open('./res_data/{}_pnl.pkl'.format(name), 'wb') as f:
#             pickle.dump(pnl[name], f)

#         with open('./res_data/{}_history.pkl'.format(name), 'wb') as f:
#             pickle.dump(history[name], f)

#         price = hedger[hedger_key].price(derivative[d_key], n_paths=N_PATHS * SAMPLE_ROUNDS) * 100
#         print(name, price)

# '''
# TRANSFORMER
# '''
# model['tf'] = Transformer(in_features=len(inputs))
# for v in ['50', '99']:
#     hedger[f'tf_es{v}'] = Hedger(
#         model['tf'], 
#         inputs=inputs,
#         sequence_prediction=True,
#         criterion=expected_shortfall[v]
#     ).to(DEVICE)

# for hedger_key in ['tf_es50', 'tf_es99']:
#     for d_key in derivative.keys():
#         name = '{}_{}'.format(hedger_key, d_key)
#         history[name] = hedger[hedger_key].fit(derivative[d_key], n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

#         pnl[name] = hedger[hedger_key].compute_pnl(derivative[d_key], n_paths=N_PATHS)
#         with open('./res_data/{}_pnl.pkl'.format(name), 'wb') as f:
#             pickle.dump(pnl[name], f)

#         with open('./res_data/{}_history.pkl'.format(name), 'wb') as f:
#             pickle.dump(history[name], f)

#         price = hedger[hedger_key].price(derivative[d_key], n_paths=int(N_PATHS * SAMPLE_ROUNDS / 20)) * 100
#         print(name, price)


# inputs.append("prev_hedge")

# '''
# MLP_withpast
# '''
# model['mlp_withpast'] = MultiLayerPerceptron(in_features=len(inputs))
# for v in ['50', '99']:
#     hedger[f'mlp_withpast_es{v}'] = Hedger(
#         model['mlp_withpast'], 
#         inputs=inputs,
#         sequence_prediction=False,
#         criterion=expected_shortfall[v]
#     ).to(DEVICE)

# for hedger_key in ['mlp_withpast_es50', 'mlp_withpast_es99']:
#     for d_key in derivative.keys():
#         name = '{}_{}'.format(hedger_key, d_key)
#         history[name] = hedger[hedger_key].fit(derivative[d_key], n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

#         pnl[name] = hedger[hedger_key].compute_pnl(derivative[d_key], n_paths=N_PATHS)
#         with open('./res_data/{}_pnl.pkl'.format(name), 'wb') as f:
#             pickle.dump(pnl[name], f)

#         with open('./res_data/{}_history.pkl'.format(name), 'wb') as f:
#             pickle.dump(history[name], f)

#         price = hedger[hedger_key].price(derivative[d_key], n_paths=N_PATHS * SAMPLE_ROUNDS) * 100
#         print(name, price)


'''
RNN
'''
model['rnn'] = ElmanRNN(in_features=len(inputs))
for v in ['50', '99']:
    hedger[f'rnn_es{v}'] = Hedger(
        model['rnn'], 
        inputs=inputs,
        sequence_prediction=True,
        criterion=expected_shortfall[v]
    ).to(DEVICE)

for hedger_key in ['rnn_es50', 'rnn_es99']:
    for d_key in derivative.keys():
        name = '{}_{}'.format(hedger_key, d_key)
        history[name] = hedger[hedger_key].fit(derivative[d_key], n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

        pnl[name] = hedger[hedger_key].compute_pnl(derivative[d_key], n_paths=N_PATHS)
        with open('./res_data/{}_pnl.pkl'.format(name), 'wb') as f:
            pickle.dump(pnl[name], f)

        with open('./res_data/{}_history.pkl'.format(name), 'wb') as f:
            pickle.dump(history[name], f)

        price = hedger[hedger_key].price(derivative[d_key], n_paths=int(N_PATHS * SAMPLE_ROUNDS / 20)) * 100
        print(name, price)



# # model = ElmanRNN(in_features=len(inputs))
# model = MultiLayerPerceptron(in_features=len(inputs))
# # model = Transformer(in_features=len(inputs))

# hedger = Hedger(
#     model, 
#     inputs=inputs,
#     sequence_prediction=False,
#     criterion=expected_shortfall
# ).to(DEVICE)

# # print(model)
# history = hedger.fit(derivative, n_epochs=N_EPOCHS, n_paths=N_PATHS, n_times=20)

# pnl = hedger.compute_pnl(derivative, n_paths=N_PATHS)
# name = 'HESTON_MLP'
# with open('./res_data/{}_pnl.pkl'.format(name), 'wb') as f:
#     pickle.dump(pnl, f)

# with open('./res_data/{}_history.pkl'.format(name), 'wb') as f:
#     pickle.dump(history, f)

# price = hedger.price(derivative, n_paths=N_PATHS * 200) * 100
# print(price)

