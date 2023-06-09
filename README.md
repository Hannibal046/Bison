## 环境依赖
```bash
git clone git@github.com:Hannibal046/Bison-Thesis.git
cd Bison-Thesis
pip install -e .
pip install transformers
```

测试: `python test.py`

## ToDo
Hannibal046:
- [x] 修改`Hedger`的数据格式
 - 在`Hedger`初始化的时候加入了一个flag,`sequence_prediction`
 - https://github.com/Hannibal046/Bison/blob/2884b9e4985a4007b3d9ed26ad49ae83ee48bb8a/pfhedge/nn/modules/hedger.py#L307-L320
- [x] 增加`RNN`模型: https://github.com/Hannibal046/Bison/blob/main/pfhedge/nn/modules/rnn.py
- [x] 增加`Transformer`模型: https://github.com/Hannibal046/Bison/blob/main/pfhedge/nn/modules/transformer.py
- [ ] 在`MLP`模型中加入多个历史信息

Bison:
- [x] 在`test.py`下面加入一段代码,测试模型的性能指标.(仅看loss,模型似乎在mlp下训练不行?)
- [ ] 加入非模拟的数据
- [ ] 跑实验

## Background

- 【期权】期权是一种衍生品，它是一种权力，围绕它的标的产生：如果标的是股票S，那简单的期权可以是在比如4月30日的时候，允许你用价格K买入股票S，那么如果那个时候S的价格高于K，你会选择行使这个权利，如果低于K，你会选择不行使这个权利；所以期权的到期价值用4.30的股票价格来看的话是个ReLU，所以期权现在的价值就由那个时候的股价决定，所以要对股票价格建模来估计期权现在的价值；

- 【期权对冲】那么期权的价格是一个跟很多东西有关的函数，包括波动率、价格、利率等等，假设是q=f(S, ...)，总体而言和S是呈现出线性性的，虽然不是完全线性，那么为了对于售出期权的那一方，会选择用股票来对冲自己的风险，那就需要售出1份期权，买入δ（一般来说0<δ<1）份股票（对应标的）来对冲掉线性部分的风险；

- 【论文观点】历史做期权定价、对冲的人，都是通过期权定价模型来做的，很数学很stochastic，但是现在他们发现直接堆DL就完事了，因为市场上的噪音不小，说不定还会比数学模型要表现的更好，所以这篇文章就是以学δ为目标，对于一个T期的期权、股票组合，学习每一期的δ就成为了任务目标，这也是为什么要生成股票序列作为sample，因为期权的存在衍生于股票

- 【期权的例子】比如说现在股票价格是30块，我在现在给你一个权利，使得4.30的时候你可以以35块买入股票，那假设4.30的时候只有两种可能，股票50%到40，50%到30，那对于这个期权的价值来说，就是50%*（40-35）=2.5，也就是我现在会以一个risk-neutral的价格2.5卖给你这份权利
