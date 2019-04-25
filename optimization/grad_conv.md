# 梯度下降类方法总结

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

对于一族有监督统计学习模型$$\mathcal{M}_\theta$$（模型的参数为$$\theta\in\Theta$$），设输入样本-标签对$$(x, y)\in\mathcal{D}$$满足$$x\in\mathcal{X}, y\in\mathcal{Y}$$，模型的决策函数（泛函）为$$f_\theta:\mathcal{X}\rightarrow\mathcal{Y}$$在$$\mathcal{X}$$上连续。给定连续的损失函数$$l:\mathcal{Y}\times\mathcal{Y}\rightarrow\mathbb{R}$$，以及定义在$$\mathcal{D}$$上的概率度量$$P$$（密度函数为p），一个模型的好坏由期望损失给出：


$$
L(M_\theta)\triangleq \mathbb{E}_{(x, y)\in\mathcal{D}}[l(f_\theta(x), y)]
$$


如果我们拿不到输入空间$$\mathcal{D}$$（需要掌握所有可能的数据生成的方式，但是我们如果有这个生成方式还训练什么模型呢？直接查表不好吗？）和概率度量$$P$$，则这个期望只能由我们已有的数据集$$D\subset\mathcal{D}$$来近似，此时在$$D$$上的损失叫做经验损失：


$$
\hat{L}(M_\theta)\triangleq \frac{1}{|D|}\sum_{(x, y)\in D}l(f_\theta(x), y),
$$


这里假设了每个样本都是服从$$P$$，从$$\mathcal{D}$$中独立抽样出来的。则梯度下降\(GD\)的更新策略：


$$
\begin{align}
\theta_{k+1} &= \theta_k - \alpha \nabla l (\theta_k)\\
&=\theta_k - \alpha\cdot\frac{1}{D}\sum_{(x, y)\in D}\frac{\partial l(f_{\theta_k}(x), y)}{\partial \theta_k}.
\end{align}
$$


当$$|D|$$很大时计算$$\nabla l(\theta_k)$$的开支较大，随机梯度下降（SGD）对$$D$$进行采样，然后用带有随机性的梯度代替梯度下降中的$$\nabla l(\theta_k)$$。记采样得到的mini-batch为$$D_k\subset D$$，随机梯度定义为


$$
\nabla l_{D_k}(\theta_k) = \frac{1}{|D_k|}\sum_{(x, y)\in D_k}\frac{l(f_{\theta_k}(x), y)}{\partial \theta_k}.
$$


由于任意一个样本$$(x, y)$$被采样出的概率为$$1/|D|$$，可以很简单地证明随机梯度的期望是经验梯度$$\mathbb{E}_{D_k}[l_{D_k}(\theta_k)]=l(\theta_k)$$。

## 收敛性分析

### Gradient Descent

对$$l(\theta)$$没有任何假设的情况下，设$$\theta^\star$$使$$l(\theta)$$取得最小值，有：


$$
\begin{align}
l(\theta_{k+1})-l(\theta^\star) &= l(\theta_{k+1})-l(\theta_k)+l(\theta_k)-l(\theta^\star)\\
&=\left\{l(\theta_k)-\alpha\nabla l(\theta_k)^T\nabla l(\theta_k)+\frac{\alpha^2}{2}\nabla l(\theta_k)^T\nabla^2 l(c)\nabla l(\theta_k)\right\}-l(\theta_k)+l(\theta_k)-l(\theta^\star),
\end{align}
$$

这里$$c$$是$$\theta_k$$和$$\theta_{k+1}$$线段上一点。如果$$\nabla^2l(c)\rightarrow\infty$$，则$$\forall \alpha > 0, l(\theta_{k+1})-l(\theta^\star) >l(\theta_k)-l(\theta^\star)$$，因此第$$k$$步迭代并没有降低损失。我们可以构造一个函数，使得从某个起点$$\theta_0$$开始，每一步梯度下降都是发散的。因此我们需要限制$$\nabla^2l(\theta)\preceq MI$$，即$$l(\theta)$$是$$M$$-Lipschitz smooth的。代入smooth条件有

$$
l(\theta_{k+1})-l(\theta^\star)
\leq\left\{\frac{\alpha^2M}{2}-\alpha\right\}\|\nabla l(\theta_k)\|^2+l(\theta_k)-l(\theta^\star),
$$

等式右侧对$$\alpha$$求最小，得$$\alpha=1/M$$时

$$
l(\theta_{k+1})-l(\theta^\star)
\leq -\frac{1}{2M}\|\nabla l(\theta_k)\|^2+l(\theta_k)-l(\theta^\star).
$$

因此当$$0<\alpha<2/M$$时，我们都有$$l(\theta_{k+1})-l(\theta^\star) < l(\theta_k)-l(\theta^\star)$$.