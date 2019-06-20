# Markov, Chebyshev and Chernoff不等式

后两个都是Markov's bound的推广。

## Markov不等式

假设随机变量$Y>0$，则$\forall\lambda >0$我们有
$$
\tag{Markov's inequality}
P\{Y\geq t\}\leq \frac{\mathbb{E}Y}{t}
$$
证明比较简单：
$$
\frac{\mathbb{E}Y}{t}=\frac{\mathbb{E}[Y\mathbf{1}_{\{Y\leq t\}}] + \mathbb{E}[Y\mathbf{1}_{\{Y\geq t\}}]}{t} \geq \frac{\mathbb{E}[Y\mathbf{1}_{\{Y\geq t\}}]}{t} \geq \frac{\mathbb{E}[t\mathbf{1}_{\{Y\leq t\}}]}{t}=P\{Y\geq t\}
$$

## Chebyshev不等式

在Markov不等式中，可以增大集合$\{Y\geq t\}$再利用Markov不等式。一种方式是通过对$Y$和$t$施加一个非负不减的函数$\phi$。取$Z=|Y-\mathbb{E}Y|$，$\phi(x)=x^2$有Chebyshev不等式：
$$
\tag{Chebyshev's inequality}
P\{|Y-\mathbb{E}Y|\geq t\} \leq \frac{Var(Y)}{t^2}.
$$

## Chernoff不等式

在Markov不等式中，取$\phi(x)=e^{\lambda x}$，其中$\lambda > 0$，有
$$
\tag{Chernoff's inequality}
P\{Y\geq t\} \leq \inf_{\lambda \geq 0}\frac{\mathbb{E}[e^{\lambda Y}]}{e^{\lambda t}}
$$
因为$P\{Y\geq t\}=P\{e^{\lambda Y}\geq e^{\lambda t}\}, \forall Y \in \mathbb{R}$，所以Chernoff不等式中$Y$是可以取整个实数域的。

注意右侧出现了矩母函数$\mathbb{E}[e^{\lambda Y}]$，因为有对数矩母函数有线性性质，我们有时候会对右侧取对数再取指数，以造出对数矩母函数$\log \mathbb{E}[e^{\lambda Y}]$：
$$
\begin{aligned}
\frac{\mathbb{E}[e^{\lambda Y}]}{e^{\lambda t}} &= \exp\left\{-\psi^\star_Y(t)\right\},
\end{aligned}
$$
其中*克莱默变换* $\psi^\star_Y(t) =\sup_{\lambda\geq 0}(\lambda t - \log \mathbb{E}[e^{\lambda Y}])$ 中出现了对数矩母。

当$t>\mathbb{E}Y$时，我们可以将对$\lambda$求最值的范围从非负数扩展到整个实数，因为此时当$\lambda<0$有
$$
\frac{\mathbb{E}[e^{\lambda Y}]}{e^{\lambda t}} \geq e^{\lambda\left(\mathbb{E}Y-t\right)}>1
$$
是恒大于左边的概率的。