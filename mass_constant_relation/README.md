\# 质量与基本常数关联性分析



本代码仓库复现了研究报告中关于粒子质量与精细结构常数α精确关联的全部计算。



\## 核心发现

1\. 质子-电子质量比公式：

&nbsp;  μ = m\_p/m\_e = 3α^{-1/2} + 4πα^{-1} + 3α^{-2/3}

&nbsp;  理论值与实验值相对误差约 0.0409%。



2\. 轻子质量谱经验公式：

&nbsp;  m\_n = A × n^B × exp(C×n)， 对 e, μ, τ 实现高精度拟合。



\## 文件说明

\- `constants.py`: 物理常数定义

\- `proton\_mass\_formula.py`: 核心公式计算

\- `lepton\_mass\_fit.py`: 轻子质量拟合

\- `plot\_results.py`: 生成分析图表

\- `main.py`: 主运行入口



\## 快速开始

1\. 安装依赖：`pip install -r requirements.txt`

2\. 运行所有计算：`python main.py`



\## 作者

\[王磊旺]

