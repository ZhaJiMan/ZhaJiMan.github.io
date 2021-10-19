---
title: "为什么样本方差的分母是 n - 1"
date: 2020-08-02
math: true
showToc: true
draft: true
tags:
- 统计
---

统计学课本的笔记，解释为什么计算样本方差时分母是 n - 1 而非 n。

<!--more-->

## 期望和方差

随机变量 $X$ 服从某种分布，定义其期望与方差分别为

$$
\begin{aligned}
E(X) &= \mu \\
D(X)
&= E[(X-\mu)^2] = \sigma^2
\end{aligned}
$$

实践中我们只能通过抽样的结果来估计 $\mu$ 和 $\sigma^2$。设样本大小为 $n$，相当于有独立同分布的随机变量 $X_1$，$X_2$，···，$X_n$，样本值为 $x_1$，$x_2$，···，$x_n$。

## 期望的估计

由于期望蕴含着随机变量的平均水平之意，所以我们会很自然地想到用样本平均值来估计期望

$$
\overline{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

从随机变量的角度来看的话，样本平均值也会随每次抽样结果的不同而随机跳动，所以样本平均值也是一个随机变量，可以记作

$$
\overline{X} = \frac{1}{n} \sum_{i=1}^{n} X_i
$$

接下来证明这一估计是无偏的。所谓无偏，即 $\overline{X}$ 的期望就等于 $\mu$——这意味着 $\overline{X}$ 的值围绕着一个正确的值上下跳动

$$
\begin{aligned}
E(\overline{X})
&= \frac{1}{n} \sum_{i=1}^{n} E(X_i) \\
&= \frac{1}{n} [nE(X)] \\
&= E(X) = \mu
\end{aligned}
$$

并且一个好的估计量应该取值稳定，即跳动程度很小。这一点可以用方差来衡量

$$
\begin{aligned}
D(\overline{X})
&= \frac{1}{n^2} \sum_{i=1}^{n} D(X_i) \\
&= \frac{1}{n^2} [nD(X)] \\
& = \frac{\sigma^2}{n}
\end{aligned}
$$

显然，样本数 $n$ 越大，这一估计越稳定。

## 方差的估计

同样的，我们自然会想到用样本方差来估计 $\sigma^2$。但是值得注意的是，大学教材中的样本方差的定义为

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X})^2
$$

这也是一个随机变量。并且，其中分母部分为 $n-1$，这便是本文标题中给出的疑惑。下面证明这一定义的 $S^2$ 是正确的——即无偏的

$$
\begin{aligned}
S^2
&= \frac{1}{n-1} \sum_{i=1}^{n} (X_i - \overline{X})^2 \\
&= \frac{1}{n-1} \sum_{i=1}^{n} (X_i^2 + \overline{X}^2 - 2X_i\overline{X}) \\
&= \frac{1}{n-1} (\sum_{i=1}^{n} X_i^2 - n\overline{X}^2) \\
&= \frac{n}{n-1} (X^2 - \overline{X}^2) \\
 \\
E(S^2)
&= \frac{n}{n-1} [E(X^2) - E(\overline{X}^2)] \\
\end{aligned}
$$

其中

$$
\begin{aligned}
E(X^2) &= \sigma^2 + \mu^2 \\
E(\overline{X}^2)
&= D(\overline{X}) + [E(\overline{X})]^2 \\
&= \frac{\sigma^2}{n} + \mu^2
\end{aligned}
$$

那么

$$
\begin{aligned}
E(S^2)
&= \frac{n}{n-1} [E(X^2) - E(\overline{X}^2)] \\
&= \frac{n}{n-1} (\sigma^2 + \mu^2 - \frac{\sigma^2}{n} - \mu^2) \\
&= \frac{n}{n-1} (\frac{n-1}{n}\sigma^2) \\
&= \sigma^2
\end{aligned}
$$

说明采用 $n-1$ 为分母的 $S^2$ 才是无偏的估计。而我们通常采用的

$$
\frac{1}{n} \sum_{i=1}^{n} (X_i - \overline{X})^2
$$

是有偏估计，会低估实际的 $\sigma^2$。当然，在 $n$ 很大的情况下，这一误差的影响可以忽略不计，两个公式都可以用来进行估计。

除了理论上的有偏无偏之外，还可以用自由度的概念来解释。如果已知期望 $\mu$，那么易证

$$
E[\frac{1}{n} \sum_{i=1}^{n} (X_i - \mu)^2] = \sigma^2
$$

即分母为 $n$ 的版本是无偏的。当期望未知时，我们会用 $\overline{X}$ 替代 $\mu$。原先的自由度为 $n$，但引入 $\overline{X}$ 后，相当于又重新利用了这 $n$ 个样本值，真正的自由度就从 $n$ 降为 $n-1$，使得上式的估计有偏。此时需要修正估计量的式子——即把分母的 $n$  改为 $n-1$。

## 标准差的估计

我们通常直接用 $S$ 来估计标准差 $\sigma$。但是，一般来说 $S$ 并不是无偏估计量。例如，对于正态总体来说，可以证明

$$
E[\frac{\Gamma(\frac{n-1}{2})\sqrt{n-1}}{\Gamma(\frac{n}{2})\sqrt{2}}] S = \sigma
$$

由于比较复杂，所以这里不再考虑。

## 极大似然估计得到的方差

如果已知一个分布的类型，但不清楚具体的参数值，可以通过极大似然估计来确定参数。例如对于正态总体，其参数恰好就是 $\mu$ 和 $\sigma$。令 $\delta=\sigma^2$，通过极大似然估计可以证明

$$
\hat{\mu} = \overline{x} \\
\hat{\delta} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \overline{x})^2
$$

这说明，极大似然估计得到的方差并不一定总是无偏的。

## Python 中函数的分母

Python 中用于计算方差的函数是 `numpy.var`，其中通过 `ddof` 参数指定分母为 `N-ddof`，且 `ddof` 默认为 0。`numpy.std` 的设置类似。但 `scipy.stats` 模块中的函数的`ddof`有时默认为 0，有时为 1，这需要查阅文档来注意。

## 参考

概率统计讲义（第三版），陈家鼎等。
[为什么样本方差（sample variance）的分母是 n-1？](https://www.zhihu.com/question/20099757/answer/658048814)