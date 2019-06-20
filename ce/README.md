# 聚集不等式

这部分笔记来自对Boucheron的《Concentration Inequalities》一书。初读这本书的门槛我感觉蛮高的，他也有一篇长论文讲聚集不等式。

聚集不等式是指的对一组独立的随机变量的某个函数引入的bound。换句话说就是bound住一组随机变量的“fluctuation”。例如，Chebyshev不等式给出了随机变量偏离其均值的概率的上界：
$$
P\{|Z-\mathbb{E}Z|\geq t\} \leq \frac{Var(Z)}{t^2}
$$
