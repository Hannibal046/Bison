## 环境依赖
```bash
git clone git@github.com:Hannibal046/Bison-Thesis.git
cd Bison-Thesis
pip install -e .
```

测试: `python test.py`

## ToDo
Hannibal046:
- [x] 修改`Hedger`的数据格式
- [x] 增加`RNN`模型
- [ ] 增加`Transformer`模型
- [ ] 在`MLP`模型中加入多个历史信息

Bison:
- [ ] 在`test.py`下面加入一段代码,测试模型的性能指标.(仅看loss,模型似乎在mlp下训练不行?)
- [ ] 加入非模拟的数据
- [ ] 跑实验