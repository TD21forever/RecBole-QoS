### 基本说明


本项目基于Recbole,适配QoS预测任务.其核心在于将QoS常用数据集WSDREAM,转换为Recbole的"原子文件",具体代码参考Recbole官网以及data文件夹中相关代码实现


### 如何使用?

- 环境设置: python=3.9.18, pip install -r requirements.txt
- 在models/models 文件夹中实现相关模型
- 在properties文件夹中定义参数,包含三部分: 数据及相关参数(sample.yaml),模型相关参数(如NeuMF.yaml),其余参数(overall.yaml)
- 以执行NeuMF为例, python test.py

### 参考资料

[Recbole 官网](https://recbole.io/cn/index.html)
[Recbole GNN](https://github.com/RUCAIBox/RecBole-GNN)