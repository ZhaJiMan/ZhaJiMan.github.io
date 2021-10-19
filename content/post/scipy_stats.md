---
title: "scipy.stats 模块的简单用法"
date: 2020-08-05
math: true
showToc: true
draft: true
tags:
- 统计
- scipy
---

## 简介

`scipy.stats` 提供了简单的统计学函数，例如随机变量的生成、计算相关系数，还有进行统计检验等。下面简单介绍如何使用该模块生成连续型随机变量，以及一些常用的方法。

<!--more-->

## 连续型随机变量

连续型随机变量通过 `rv_continuous` 类表示，常用的分布有

- `stats.norm`：正态分布。
- `stats.chi2`：卡方分布。
- `stats.t`：t 分布。
- `stats.f`: F 分布。

例如要生成一个标准正态分布的随机变量，操作为

```Python
from scipy import stats
rv = stats.norm(loc=0, scale=1)
```

其中参数 `loc` 表示分布的位置，`scale` 表示对分布的放缩。这两个参数对于不同的分布有不同的意义，需要查询文档来确定。例如对正态分布来说，`loc` 就是 $\mu$，`scale` 就是 $\sigma$。默认值为 `loc=0`，`scale=1`，所以其实 `stats.norm()` 就能生成一个标准正态分布。

除了考虑平移和放缩的这两个参数外，某些分布还需要更多的参数，例如 t 分布需要给出自由度 `df`。

求分布的统计量有两种风格，一是直接调用分布的函数，但每次都需要提供形状参数；二是面向对象的风格，通过调用方法进行计算。后者被称为“冻结”（freezing）一个分布，例如

```Python
# 直接调用时需要给出参数.
print(stats.norm.mean(loc=0, scale=1))

# 面向对象风格,可以反复调用.
rv = stats.norm(loc=0, scale=1)
print(rv.mean())
```

## 常用方法

```Python
# 生成n个服从该分布的样本.
rv.rvs(size=n)

# 计算该分布的一些统计量.'mv'指示平均值和方差.
rv.stats(moments='mv')

# 得到数组x所对应的PDF函数.
rv.pdf(x)

# 得到数组x所对应的CDF函数.
rv.cdf(x)

# 得到数组x所对应的Survival Function,即1-CDF.
rv.sf(x)

# 得到数组q所对应的Percent Point Function,即CDF的逆函数.
rv.ppf(x)

# 得到数组q所对应的SF逆函数.
rv.isf(q)
```

下面用图示说明一下 `ppf` 和 `isf` 函数的用法。

![quantile1](/scipy_stats/quantile1.png)

如上图所示，对于标准正态分布，设显著性水平 $\alpha=0.05$，`ppf(1-alpha)` 能够给出左半部分累计概率为 $95\%$ 时对应的 $x$ 的值，即下分位点。

![quantile2](/scipy_stats/quantile2.png)

如上图所示，`isf(1-alpha)` 能够给出右半部分累计概率为 $95\%$ 时对应的 $x$ 的值，即上分位点。等价的表述为`ppf(alpha)`。

![quantile3](/scipy_stats/quantile3.png)

如上图所示，通过 `isf(1-alpha/2)` 和 `ppf(1-alpha/2)`（或者 `isf(alpha/2)` 和 `ppf(alpha/2)`），可以找出中间部分累计概率为 $95\%$ 时对应的两个 $x$ 的值：$\pm1.96$。

画图的代码如下

```Python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def draw_plot(alpha, side):
    '''画出正态分布的分位数图.
    
    side=left表示下alpha分位点,右半部分概率为1-alpha.
    side=right表示上alpha分位点,左半部分概率为1-alpha.
    side=both表示上下alpha/2分位点,中间部分概率为1-alpha.
    '''

    rv = stats.norm(loc=0, scale=1)
    x = np.linspace(-5, 5, 1000)
    y = rv.pdf(x)

    # 计算分位点
    if side == 'left':
        xp = rv.isf(1-alpha)
        flag = (x >= xp)
        yp = rv.pdf(xp)
    elif side == 'right':
        xp = rv.ppf(1-alpha)
        flag = (x <= xp)
        yp = rv.pdf(xp)
    elif side == 'both':
        xpl = rv.isf(1-alpha/2)
        xpr = rv.ppf(1-alpha/2)
        flag = (x >= xpl) & (x <= xpr)
        yp = rv.pdf(xpl)

    xf = x[flag]
    yf = y[flag]

    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111)

    # 画出PDF曲线和1-alpha概率的区间
    ax.plot(x, y)
    ax.fill_between(xf, yf, alpha=0.5)

    # 标出概率
    ax.text(0, y.max()/2.5, f'{(1-alpha)*100}%', fontsize='large', \
            ha='center', va='center')
    
    # 标出分位点
    if side == 'left':
        ax.plot(xp, yp, 'ko', ms=3)
        ax.text(1.05*xp, 1.05*yp, f'x={xp:.2f}', ha='right')
    elif side == 'right':
        ax.plot(xp, yp, 'ko', ms=3)
        ax.text(1.05*xp, 1.05*yp, f'x={xp:.2f}', ha='left')
    elif side == 'both':
        ax.plot([xpl, xpr], [yp, yp], 'ko', ms=3)
        ax.text(1.05*xpl, 1.05*yp, f'x={xpl:.2f}', ha='right')
        ax.text(1.05*xpr, 1.05*yp, f'x={xpr:.2f}', ha='left')

    ax.set_ylim(0, None)
    ax.set_xlabel('x', fontsize='large')
    ax.set_ylabel('PDF', fontsize='large')
    ax.set_title('Normal Distribution', fontsize='large')

    fig.savefig('normal_' + side)
    plt.close(fig)

alpha = 0.05
draw_plot(alpha, side='left')
draw_plot(alpha, side='right')
draw_plot(alpha, side='both')
```