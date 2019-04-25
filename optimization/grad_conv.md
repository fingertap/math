# Gradient Descent Convergence Analysis

这篇里记录一下梯度下降在一般条件下的收敛分析。关键的思想有
1. 利用泰勒公式+拉格朗日余项对函数做二阶展开。
2. 收敛$$\Leftrightarrow f$$是Lipschitz smooth。
3. 收敛速度关键在于对$$\|\nabla f(x)\|$$引入下界。

## 基础

考虑可导函数$$f:\mathbb{R}^d\rightarrow\mathbb{R}$$，在任一点处展开有：
$$
f(x) = f(y) + (x-y)^T\nabla f(y) + (x-y)^T\nabla^2f(c)(x-y),
$$
其中$$c$$是$$x$$和$$y$$线段上一点。我们说他是**凸函数**，意味着$$\forall c\in\mathbf{dom}f, \nabla^2f(c)\succeq0$$，即$$\forall x, y\in \mathbf{dom}f$$
$$
f(x) \geq f(y) + (x-y)^T\nabla f(y).
$$

Lipschitz smooth和Strong convexity类似，是对$$\nabla^2f(c)$$引入了上下界。假设$$f(x)$$是$$M$$-Lipschitz smooth，以及$$m$$-strongly convex，有$$\forall c\in\mathbf{dom}f$$
$$
mI\preceq \nabla^2f(c)\preceq MI,
$$
其中$$I$$是单位矩阵。注意这样一来Convex可以看做是0-strongly convex。Lipschitz smooth直觉理解就是没有折点（例如$$|x|$$在$$x=0$$处），Strong convexity直觉理解就是没有盆地（一片区域的函数值相等）。

对于一族有监督统计学习模型$$\mathcal{M}_\theta$$（模型的参数为$$\theta$$），设输入样本-标签对$$(x, y)\in\mathcal{D}$$满足$$x\in\mathcal{X}, y\in\mathcal{Y}$$，模型的决策函数（泛函）为$$f_\theta:\mathcal{X}\rightarrow\mathcal{Y}$$在$$\mathcal{X}$$上连续。给定连续的损失函数$$l:\mathcal{Y}\times\mathcal{Y}\rightarrow\mathbb{R}$$，以及定义在$$\mathcal{X}$$上的概率度量$$P$$（密度函数为p），一个模型的好坏由期望误差损失给出：
$$
L(M_\theta)\triangleq
\mathbb{E}
_{(x, y)\in\mathcal{D}}[l(f_\theta(x), y)]
$$
则梯度下降(GD)的更新策略：
$$
\theta_{k+1} = \theta_k - \alpha \nabla f(\theta_k)
$$
其中$$\nabla f(\theta_k)$$由经验
## 收敛性分析

### Gradient Descent

