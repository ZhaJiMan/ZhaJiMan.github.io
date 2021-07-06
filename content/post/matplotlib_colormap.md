---
title: "Matplotlib 系列：colormap 的设置"
date: 2021-07-05
math: true
showToc: true
tags:
- matplotlib
---

## 0. 前言

所谓 colormap（颜色表），就是将一系列颜色按给定的顺序排列在一起。其用处是，我们可以通过某种映射关系，将一系列数值映射到一张 colormap 上去，使不同大小的数值对应不同的颜色。这样一来，在绘制填色图时便能直观地用颜色来反映数值的分布。

在 Matplotlib 中，数值到颜色的映射关系可以用下面这张图来表示

![value_color_mapping.png](/matplotlib_colormap/value_color_mapping.png)

图中分为前后两部分

- 首先将数组的数值归一化（Normalization）到浮点型的 [0.0, 1.0] 范围或整型的 [0, N - 1] 范围上去。
- 再把归一化的数据输入给 colormap，输出数组数值对应的颜色（RGBA 值）。

第二部分的映射关系基本上是固定不变的，但第一部分的映射关系可以通过 Matplotlib 的许多类来加以改变，进而实现对数色标、对称色标、离散色标等一系列可视化效果。

本文将会依次介绍 `Colormap` 类、归一化会用到的类，以及实际应用的例子。

<!--more-->

## 1. Colormap

很容易想到，一系列颜色可以用 N * 4 大小的 RGBA 数组表示。但是 matplotlib 中的 colormap 并非简单的数组，而是专门用一个 `Colormap` 类实现的，有着更加方便的重采样功能。内置的所有 colormap 存放在 `matplotlib.cm` 模块下，它们的名字在官网的 [Choosing Colormaps in Matplotlib](https://matplotlib.org/3.3.3/tutorials/colors/colormaps.html#sphx-glr-tutorials-colors-colormaps-py) 页面中可以找到。

`Colormap` 有两个子类：`ListedColormap` 和 `LinearSegmentedColormap`，它们被存放在 `matplotlib.colors` 模块下。下面来分别介绍它们。

### 1.1 ListedColormap

顾名思义，将所有颜色列举到一个列表中，便能生成这一类的 colormap。一个简单的例子如下

```Python
import matplotlib as mpl
import matplotlib.pyplot as plt

cmap = mpl.colors.ListedColormap(
    ["darkorange", "gold", "lawngreen", "lightseagreen"]
)
```

列表中的元素可以是 RGBA 值，也可以是颜色的名字。这个 colormap 看起来是这样的

![ListedColormap_1](/matplotlib_colormap/ListedColormap_1.png)

正好是我们放入列表中的四种颜色。

`cmap.colors` 是这个 colormap 的所有颜色的 RGBA 值组成的元组，而 `cmap.N` 是颜色的总数，显然这里 N = 4。`cmap` 对象可以用数值参数调用，返回数值对应的颜色 RGBA 值，根据数值是整型还是浮点型，对应关系也会有所不同，如下图所示

![mapping](/matplotlib_colormap/mapping.png)

当参数 x 为整数时，对应于第 x - 1 个颜色；当参数 x 为浮点数时，返回它落入的区间对应的颜色。当参数 x 超出 [0, N-1] 或 [0.0, 1.0] 的范围时，对应于第一个和最后一个颜色。这一特性能让我们很简单地索引 colormap 中的颜色，例如

![color_indexing](/matplotlib_colormap/color_indexing.png)

可以看到用不同类型的参数索引出的 RGBA 数组是一致的。再举个利用索引结果创建新 colormap 的例子

```Python
cmap_new = mpl.colors.ListedColormap(
    cmap(np.linspace(0, 1, 5))
)
```
`cmap_new` 看起来会是这个样子

![ListedColormap_2](/matplotlib_colormap/ListedColormap_2.png)

因为给出的参数中，最后两个数落进了同一个区间，所以对应的颜色相同。

### 1.2 LinearSegmentedColormap

顾名思义，是通过线性分段构建的 colormap。首先给定几个颜色的锚点，然后锚点之间的颜色会通过线性插值得出。直接初始化该类的方法比较难以理解，所以一般会用 `LinearSegmentedColormap.from_list` 函数来创建对象，有需求的读者可以参阅文档。

Matplotlib 中大部分 colormap 都属于 `LinearSegmentedColormap`，例如常用的 `jet`

```Python
cmap = mpl.cm.jet
```

看起来是这样的

![LinearSegmentedColormap_1](/matplotlib_colormap/LinearSegmentedColormap_1.png)

与 `ListedColormap` 相比，`LinearSegmentedColormap` 依旧有 `cmap.N` 属性，默认数值为 256。但是没有了 `cmap.colors`，不能直接列出这 N 个颜色的 RGBA 值。

`cmap` 依旧可以被直接调用：当参数 x 为整数时，对应于第 x + 1 个颜色；而当参数 x 为浮点数时，则会通过线性插值获取相邻两个颜色中间的颜色。因此，`LinearSegmentedColormap` 的重采样不仅不会出现重复的颜色，还能得到更为连续渐变的颜色。

### 1.3 get_cmap 函数

有时我们希望通过重采样直接得到一个新的 colormap，而不是得到一组 RGBA 值，这个需求可以用 `mpl.cm.get_cmap` 函数实现，例如对 `jet` 采样 8 个颜色

```Python
# 等价于用mpl.cm.jet(np.linspace(0, 1, 8))的结果创建LinearSegmentedColormap.
cmap = mpl.cm.get_cmap('jet', 8)
```

效果如下图。并且采样得到的 colormap 类型与被采样的保持一致。

![LinearSegmentedColormap_2](/matplotlib_colormap/LinearSegmentedColormap_2.png)

### 1.4 set_under、set_over 与 set_bad

1.1 节中提到过，直接调用 `cmap` 时，若参数 x 超出范围，那么会映射给第一个或最后一个颜色。而 `cmap.set_under` 方法能够改变 x < 0 或 x < 0.0 时对应的颜色，`cmap.set_over` 方法能够改变 x > N - 1 或 x > 1.0 时对应的颜色。`cmap.set_bad` 则能改变缺测值（`nan` 或 `masked`）对应的颜色（缺测值的绘图规则请参考之前的博文 [NumPy 系列：缺测值处理](https://zhajiman.github.io/post/numpy_missing_value)）。

使用 `fig.colorbar` 方法画 colorbar 时，通过 `extend` 参数可以指定是否在 colorbar 两端显示出 under 与 over 时的颜色。下面为一个例子

```Python
cmap = mpl.cm.get_cmap('jet', 8)
cmap.set_under('black')
cmap.set_over('white')
```

![LinearSegmentedColormap_3](/matplotlib_colormap/LinearSegmentedColormap_3.png)

## 2. Normalization

上一节的重点是，colormap 能把 [0.0, 1.0] 或 [0, N - 1] 范围内的值映射到颜色上，那么这一节就要来叙述如何通过归一化（Normalization）把原始数据映射到 [0.0, 1.0] 或 [0, N - 1] 上。用于归一化的类都存放在 `mpl.colors` 模块中。

### 2.1 Normalize

各种二维绘图函数在进行归一化时都默认使用 `Normalize` 类。给定参数 `vmin` 和 `vmax`，它会按照线性关系

$$
y=\frac{x-vmin}{vmax-vmin}
$$

将原始数据 x 映射为 y。虽然这一操作叫做“归一化”，但显然只有 [vmin, vmax] 范围内的 x 会被映射到 [0.0, 1.0] 上，其它 x 映射出来的 y 会小于 0.0 或大于 1.0。不过若是不给定 `vmin` 和 `vmax`，则默认用 x 的最小值和最大值来代替，此时所有 x 都会被映射到 [0.0, 1.0] 上。下面是一个归一化后的结果都在 [0.0, 1.0] 范围内的例子

![Normalize](/matplotlib_colormap/Normalize.png)

归一化后的值可以直接传给 colormap，以得到画图用的颜色。即便归一化后的结果超出了 [0.0, 1.0] 的范围，根据第 1 节中的说明，这些超出的值会被映射给第一个或最后一个颜色（或者 `set_under` 和 `set_over` 指定的颜色），换句话说，[vmin, vmax] 范围外的 x 自然对应于 colormap 两端的颜色。

此外，`Normalize` 还有一个 `clip` 参数，当它为 True 时，能把 [vmin, vmax] 范围外的 x 映射为 0.0 或 1.0，不过这样一来，colormap 的 under 与 over 的设置便会失去作用。所以一般我们不用关心 `clip` 参数，让它默认为 False 就好了。

### 2.2 LogNorm

类似于 `Normalize`，`LogNorm` 能将 [vmin, vmax] 范围内的 x 的对数线性映射到 [0.0, 1.0] 上，公式表示为
$$
y = \frac{\log_{10}(x) - \log_{10}(vmin)}{\log_{10}(vmax) - \log_{10}(vmin)}
$$
其中 `vmin` 和 `vmax` 必须为正数，否则会报错；x 可以小于等于 0，不过结果会缺测（`masked`）。例如

![LogNorm](/matplotlib_colormap/LogNorm.png)

除了对数关系外，Matplotlib 还提供任意幂律关系的 `PowerNorm`，此处不再详细介绍。

### 2.3 BoundaryNorm

除了线性和对数的映射，有时我们需要的映射关系像是往一组摆在一起的框里投球。例如下图这个例子

![bins_example](/matplotlib_colormap/bins_example.png)

给出一系列边缘靠在一起的 bin（框子），原始数据落入第几个框（左闭右开区间），就对应于第几个颜色。因为这些框边缘的数值可以任意给定，所以很难用简单的函数表示。为了实现这种映射，这里引入 `BoundaryNorm`。

参数 `boundaries` 为我们给出的这些 bin 的边缘数值，要求单调递增；`ncolors` 则是我们希望与之对应的 colormap 中颜色的数目（即 `cmap.N`），其数值大于等于 `nbin = len(boundaries) - 1`。

当 ncolors = nbin 时，映射关系为：
$$
y = \begin{cases}
i &\text{if} \quad boundaries[i] \le x < boundaries[i+1] \newline
-1 &\text{if} \quad x < boundaries[0] \newline
nbin &\text{if} \quad x \ge boundaries[-1]
\end{cases}
$$
可以看到，落入框中的 x 会被映射到 [0, nbin - 1] 上，而没有落入框中的 x 会映射为 -1 或 nbin。

当 ncolors > nbin 时，落入框中的 x 会被映射到 [0, ncolors - 1] 上。我觉得这种情况下的映射关系不是很直观，所以公式就不列了，平时我也会尽量避开这种情况。此外 `BoundaryNorm` 还有个 `extend` 参数，也会使映射关系复杂化，建议不要去设置它。下面举个例子

![BoundaryNorm](/matplotlib_colormap/BoundaryNorm.png)

### 2.4 CenteredNorm

这是 Matplotlib 3.4.0 新引入的归一化方法，给定对称中心 `vcenter` 和中心向两边的范围 `halfrange`，有映射关系
$$
y = \frac{x - (vcenter - halfrange)}{2 \times halfrange}
$$
意义很明确，即 `vcenter` 两边的 x 会被线性映射到 0.5 两边。由于这个类要求的 Matplotlib 版本太高，估计很多人还用不了，不过要用 `Normalize` 来实现相同的结果也很简单。

### 2.5 TwoSlopeNorm

类似于 `CenteredNorm`，也是会把 `vcenter` 两边的 x 线性映射到 0.5 两边，但是 `vcenter` 向两边延伸的范围可以不等。映射关系为
$$
y = \begin{cases}
0.0 &\text{if} \quad x < vmin \newline
(x - vmin) / (vcenter - vmin) &\text{if} \quad vmin \le x < vcenter \newline
(x - vcenter) / (vmax - vcenter) &\text{if} \quad vcenter \le x < vcenter \newline
1.0 &\text{if} \quad x \ge vmax
\end{cases}
$$
其内部是用 `np.interp` 函数完成计算的，所以超出 [vmin, vmax] 范围的 x 会被映射为 0.0 或 1.0。

## 3 实际应用

### 3.1 pcolor 和 contour 的异同

对于画马赛克图的 `pcolor`、`pcolormesh` 和 `imshow` 函数，实际使用时我们并不需要手动进行数据的归一化和颜色采样，只需在调用函数时通过 `cmap` 和 `norm` 参数把 colormap 和归一化的类传入即可，绘图函数会自动计算数据和颜色的对应关系。因为线性的归一化方法最为常用，所以这些函数都默认使用 `Normalize` 类，并默认用数据的最小最大值作为 `vmin` 和 `vmax`。下面是例子

```python
# 生成测试数据.
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = 1E3 * np.exp(-(np.abs(X - 5)**2 + np.abs(Y - 5)**2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cmap = mpl.cm.jet

# 使用默认的线性归一化,可以直接给出vmin和vmax.
im = axes[0].pcolormesh(
    X, Y, Z, cmap=cmap, vmin=Z.min(), vmax=Z.max(),
    shading='nearest'
)
cbar = fig.colorbar(im, ax=axes[0], extend='both')
axes[0].set_title('Normalize')

# 若在pcolormesh中给定了norm,就不能再指定vmin和vmax了.
norm = mpl.colors.LogNorm(vmin=1E-3, vmax=1E3)
im = axes[1].pcolormesh(
    X, Y, Z, cmap=cmap, norm=norm,
    shading='nearest'
)
# 使用LogNorm时,colorbar会自动选用_ColorbarLogLocator来设定ticks.
cbar = fig.colorbar(im, ax=axes[1], extend='both')
axes[1].set_title('LogNorm')

plt.show()
```

![application_1](/matplotlib_colormap/application_1.png)

可以看到 `LogNorm` 能让数据的颜色分布不那么集中。

而画等高线的 `contour` 和 `contourf` 则与 `pcolor` 有一些细节上的差异。这两个函数多了个 `levels` 参数，用于指定每条等高线对应的数值。它们默认使用 `Normalize(vmin=min(levels), max(levels))` 作为归一化的方法，如果我们给出了 `vmin` 和 `vmax`，则优先使用我们给出的值。对于 `contour`，每条等高线的颜色可以表示为 `cmap(norm(levels))`；对于 `contourf`，等高线间填充的颜色可以表示为

```python
# 在norm不是LogNorm的情况下,layers计算为levels的中点.详请参考matplotlib.contour模块.
levels = np.array(levels)
layers = 0.5 * (levels[1:] + levels[:-1])
colors = cmap(norm(layers))
```

`contourf` 默认不会填充 `levels` 范围以外的颜色，如果有这方面的需求，可以用 `extend` 参数指定是否让超出范围的数据被填上 colormap 两端的颜色（或 `set_under` 和 `set_over` 指定的颜色）。

举个同时画出等高线和填色图的例子，填色设为半透明

```python
# 生成测试数据.
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = (X - 5) ** 2 + (Y - 5) ** 2
# 将Z的值缩放到[0, 100]内.
Z = Z / Z.max() * 100

# 设置一个简单的colormap.
cmap = mpl.colors.ListedColormap(['blue', 'orange', 'red', 'purple'])
fig, ax = plt.subplots()
# contour和contourf默认使用levels的最小最大值作为vmin和vmax.
levels = np.linspace(10, 60, 6)
im1 = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, alpha=0.5)
im2 = ax.contour(X, Y, Z, levels=levels, cmap=cmap, linewidths=2)
cbar = fig.colorbar(im1, ax=ax)
# 为等高线添加标签.
ax.clabel(im2, colors='k')

plt.show()
```

![application_2](/matplotlib_colormap/application_2.png)

可以看到，`levels` 范围以外的部分直接露出了白色背景。等高线的颜色与等高线之间的填色并不完全一致，这是 `levels` 和 `layers` 之间的差异导致的。以上提到的这些参数都可以在 `contour` 和 `contourf` 函数返回的 `QuadContourSet` 对象的属性中找到，有兴趣的读者可以自己调试看看。

### 3.2 BoundaryNorm 的应用

直接上例子

```python
# 生成测试数据.
x = np.linspace(0, 10, 100)
y = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2
# 将Z的值缩放到[0, 100]内.
Z = Z / Z.max() * 100

# 设置norm.
bins = [1, 5, 10, 20, 40, 80]
nbin = len(bins) - 1
norm = mpl.colors.BoundaryNorm(bins, nbin)
# 设置cmap.
cmap = mpl.cm.get_cmap('jet', nbin)
cmap.set_under('white')
cmap.set_over('purple')

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
cbar = fig.colorbar(im1, ax=axes[0], extend='both')
axes[0].set_title('pcolormesh')

# 注意contourf设置extend时,colorbar就不要设置extend了.
im2 = axes[1].contourf(X, Y, Z, levels=bins, cmap=cmap, norm=norm, extend='both')
cbar = fig.colorbar(im2, ax=axes[1])
axes[1].set_title('contourf')

plt.show()
```

![application_3](/matplotlib_colormap/application_3.png)

在对 `contourf` 应用 `BoundaryNorm` 时，很容易联想到，等高线就相当于 `bins` 的边缘，等高线之间的填色正好对应于每个 bin 中的颜色，所以指定 `levels=bins` 是非常自然的。如果不这样做，`contourf` 默认会根据数据的范围，利用 `MaxNLocator` 自动生成 `levels`，此时由于 `levels` 与 `bins` 不匹配，填色就会乱套。

### 3.3 红蓝 colormap

当数据表示瞬时值与长时间平均值之间的差值时，我们常用两端分别为蓝色和红色的 colormap，并将数据的负值和正值分别映射到蓝色和红色上，这样画出来的图一眼就能看出哪里偏高哪里偏低。下面分别用 `Normalize` 和 `TwoSlopeNorm` 来实现

```python
# 生成测试数据.
X, Y = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100))
Z1 = np.exp(-X**2 - Y**2)
Z2 = np.exp(-(X - 1)**2 - (Y - 1)**2)
Z = ((Z1 - Z2) * 2)
# 将Z的值缩放到[-5, 10]内.
Z = (Z - Z.min()) / (Z.max() - Z.min()) * 15 - 5

# 设定红蓝colormap与两种norm.
cmap = mpl.cm.RdBu_r
norm_list = [
    mpl.colors.Normalize(vmin=-10, vmax=10),
    mpl.colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=10)
]
# levels需要与norm的范围相匹配.
levels_list = [
    np.linspace(-10, 10, 21),
    np.linspace(-5, 10, 16)
]
# 图片需要的标题.
title_list = [
    'Normalize',
    'TwoSlopeNorm'
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i in range(2):
    im = axes[i].contourf(
        X, Y, Z, levels=levels_list[i], cmap=cmap,
        norm=norm_list[i], extend='both'
    )
    cbar = fig.colorbar(im, ax=axes[i])
    axes[i].set_title(title_list[i])

plt.show()
```

![application_4](/matplotlib_colormap/application_4.png)

如果你的 Matplotlib 版本够高的话，还可以试试 `CenteredNorm`。这三种归一化方法都是线性的，非线性的方法有 `SymLogNorm`，或者用 `BoundaryNorm` 也可以实现。

### 3.4 自定义归一化方法

请参考 Matplotlib 官网的 [Colormap Normalization](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html) 教程的最后一节。

## 4. 结语

以上便是对 Matplotlib 中 colormap 的简要介绍，有错误的话烦请在评论区指出。下期将会接着介绍与之密不可分的 colorbar。

## 参考链接

参考的全是 Matplotlib 官网的教程

[Customized Colorbars Tutorial](https://matplotlib.org/stable/tutorials/colors/colorbar_only.html)

[Creating Colormaps in Matplotlib](https://matplotlib.org/stable/tutorials/colors/colormap-manipulation.html)

[Colormap Normalization](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html)

如果想自定义 colormap 的话，可以参考

[Beautiful custom colormaps with Matplotlib](https://towardsdatascience.com/beautiful-custom-colormaps-with-matplotlib-5bab3d1f0e72)