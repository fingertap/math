# Sub-Gaussian and Sub-Gamma Variable

次高斯变量是一类从属于高斯变量的随机变量，高斯变量的许多性质在次高斯上也为真。Gamma变量类似。

一个比较好的次高斯变量的note可以在[这里](https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)找到。

## Sub-Gaussian

### 定义

一个实变量$$Y$$被称为$$v$$-subgaussian，若$$\forall \lambda \in \mathbb{R}$$，矩母函数被高斯矩母函数控制
$$
\tag{Sub-Gaussian}
\mathbb{E}[e^{\lambda Y}] \leq e^{\lambda^2v^2/2}.
$$

### 性质

#### 类高斯矩

若$$Y$$为$$v$$-subgaussian，则$$\mathbb{E}Y=0$$，且$$Var(Y)\leq b^2$$。

**证明**：
对$$\lambda\in\mathbb{R}$$，在$$(\text{Sub-Gaussian})$$式两侧对指数函数$$e^x$$泰勒展开

$$
\sum_{n=0}^\infty\frac{\lambda^n}{n!}\mathbb{E}[Y^n]=\mathbb{E}[e^{\lambda Y}]\leq e^{v^2\lambda^2/2} = \sum_{n=0}^\infty\left(\frac{v^2\lambda^2}{2}\right)^n\cdot\frac{1}{n!}
$$

常数1被抵消，左边取前两项：

$$
\lambda\cdot\mathbb{E}Y+\mathbb{E}[Y^2]\frac{\lambda^2}{2} \leq \sum_{n=1}^\infty\left(\frac{v^2\lambda^2}{2}\right)^n\cdot\frac{1}{n!}
$$

两边除以大于零的$$\lambda$$，并令$$\lambda\rightarrow 0$$有$$\mathbb{E}Y\leq 0$$；对$$\lambda<0$$类似有$$\mathbb{E}Y\geq 0$$，因此$$\mathbb{E}Y=0$$。代入$$\mathbb{E}Y=0$$，再除以$$\lambda^2\rightarrow 0$$，有$$\mathbb{E}[Y^2]\leq v^2$$。

#### 线性

若$$Y$$为$$v$$-subgaussian，则$$\forall \alpha\in\mathbb{R}$$，$$\alpha Y$$是$$|\alpha|v$$-subgaussian的。若$$Y_1, Y_2$$分别为$$v_1, v_2$$-subgaussian，则$$Y_1+Y_2$$为$$(v_1+v_2)$$-subgaussian。

利用定义可以容易地验证。

### 刻画

下面三种对Sub-Gaussian变量的刻画等价：

- Laplace transform (moment generating function) condition: $$\exists v>0, \forall \lambda \in \mathbb{R}$$，使得$$\mathbb{E}[e^{\lambda Y}]\leq e^{v^2t^2/2}$$；
- Subgaussian tail: $$\exists c>0, \forall t > 0$$，$$p(|Y|\geq t)\leq 2e^{-ct^2}$$；
- $$\psi_2$$-condition: $$\exists a > 0$$， $$\mathbb{E}[e^{aY^2}]\leq 2$$.

证明见[这里](https://sites.ualberta.ca/~omarr/publications/subgaussians.pdf)。直观理解就是说比高斯变量要更聚集。
