---
title: "Matplotlib 系列：colorbar 的设置"
date: 2021-07-10
showToc: true
tags:
- matplotlib
---

## 0. 前言

承接 [Matplotlib 系列：colormap 的设置](https://zhajiman.github.io/post/matplotlib_colormap/) 一文，这次介绍 colorbar。所谓 colorbar 即主图旁一个长条状的小图，能够辅助表示主图中 colormap 的颜色组成和颜色与数值的对应关系。本文将会依次介绍 colorbar 的基本用法、如何设置刻度，以及怎么为组图添加 colorbar。本文基于 Matplotlib 3.3.4。

<!--more-->

## 1. colorbar 的基本用法

Colorbar 主要通过 `figure.colorbar` 方法绘制，先介绍常用的几个参数

- `mappable`：直译为“可映射的”，要求是 `matplotlib.cm.ScalarMappable` 对象，能够向 colorbar 提供数据与颜色间的映射关系（即 colormap 和 normalization 信息）。主图中使用 `contourf`、`pcolormesh` 和 `imshow` 等二维绘图函数时返回的对象都属于 `ScalarMappable`。
- `cax`：colorbar 本质上也是一种特殊的 axes，我们为了在画布上决定其位置、形状和大小，可以事先画出一个空 axes，然后将这个 axes 提供给 `cax` 参数，那么这个空 axes 就会变成 colorbar。
- `ax`：有时我们懒得手动为 colorbar 准备好位置，那么可以用 `ax` 参数指定 colorbar 依附于哪个 axes，接着 colorbar 会自动从这个 axes 里“偷”一部分空间来作为自己的空间。
- `orientation`：指定 colorbar 的朝向，默认为垂直方向。类似的参数还有 `location`。
- `extend`：设置是否在 colorbar 两端额外标出 normalization 范围外的颜色。如果 colormap 有设置过 `set_under` 和 `set_over`，那么使用这两个颜色。
- `ticks`：指定 colorbar 的刻度位置，可以接受 ticks 的序列或 `Locator` 对象。
- `format`：指定 colorbar 的刻度标签的格式，可以接受格式字符串，例如 `'%.3f'`，或 `Formatter` 对象。
- `label`：整个 colorbar 的标签，类似于 axes 的 xlabel 或 ylabel。

此外 colorbar 还有些设置不能在初始化的时候一次性搞定，需要接着调用方法才能完成。

### 1.1 单独绘制 colorbar

虽然 colorbar 一般依附于一张填色的主图，但其实只要给出 colormap 和 normalization 就能决定 colorbar 了。下面给出单独绘制 colorbar 的例子

```python
import copy
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt

fig, axes = plt.subplots(4, 1, figsize=(10, 5))
fig.subplots_adjust(hspace=4)

# 第一个colorbar使用线性的Normalize.
cmap1 = copy.copy(mpl.cm.viridis)
norm1 = mpl.colors.Normalize(vmin=0, vmax=100)
im1 = mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1)
cbar1 = fig.colorbar(
    im1, cax=axes[0], orientation='horizontal',
    ticks=np.linspace(0, 100, 11),
    label='colorbar with Normalize'
)

# 第二个colorbar开启extend参数.
cmap2 = copy.copy(mpl.cm.viridis)
cmap2.set_under('black')
cmap2.set_over('red')
norm2 = mpl.colors.Normalize(vmin=0, vmax=100)
im2 = mpl.cm.ScalarMappable(norm=norm2, cmap=cmap2)
cbar2 = fig.colorbar(
    im2, cax=axes[1], orientation='horizontal',
    extend='both', ticks=np.linspace(0, 100, 11),
    label='extended colorbar with Normalize'
)

# 第三个colorbar使用对数的LogNorm.
cmap3 = copy.copy(mpl.cm.viridis)
norm3 = mpl.colors.LogNorm(vmin=1E0, vmax=1E3)
im3 = mpl.cm.ScalarMappable(norm=norm3, cmap=cmap3)
# 使用LogNorm时,colorbar会自动选取合适的Locator和Formatter.
cbar3 = fig.colorbar(
    im3, cax=axes[2], orientation='horizontal',
    label='colorbar with LogNorm',
)

# 第四个colorbar使用BoundaryNorm.
bins = [0, 1, 10, 20, 50, 100]
nbin = len(bins) - 1
cmap4 = mpl.cm.get_cmap('viridis', nbin)
norm4 = mpl.colors.BoundaryNorm(bins, nbin)
im4 = mpl.cm.ScalarMappable(norm=norm4, cmap=cmap4)
# 使用BoundaryNorm时,colorbar会自动按bins标出刻度.
cbar4 = fig.colorbar(
    im4, cax=axes[3], orientation='horizontal',
    label='colorbar with BoundaryNorm'
)

plt.show()
```

![colorbar_only](/matplotlib_colorbar/colorbar_only.png)

colorbar 使用的 colormap 和 normalization 的信息可以通过 `cbar.cmap` 和 `cbar.norm` 属性来获取。

### 1.2 向主图添加 colorbar

日常使用中一般不会单独画出 colorbar，而是将 colorbar 添加给一张主图。此时需要将主图中画填色图时返回的 `ScalarMappable` 对象传给 colorbar，并利用 `cax` 或 `ax` 参数指定 colorbar 的位置。下面是一个例子

```python
def add_box(ax):
    '''用红框标出一个ax的范围.'''
    axpos = ax.get_position()
    rect = mpl.patches.Rectangle(
        (axpos.x0, axpos.y0), axpos.width, axpos.height,
        lw=3, ls='--', ec='r', fc='none', alpha=0.5,
        transform=ax.figure.transFigure
    )
    ax.patches.append(rect)

def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax

def test_data():
    '''生成测试数据.'''
    x = np.linspace(-3, 3, 200)
    y = np.linspace(-3, 3, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-X**2) + np.exp(-Y**2)
    # 将Z缩放至[0, 100].
    Z = (Z - Z.min()) / (Z.max() - Z.min()) * 100

    return X, Y, Z

X, Y, Z = test_data()
cmap = mpl.cm.viridis

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
fig.subplots_adjust(hspace=0.2, wspace=0.2)

# 提前用红框圈出每个ax的范围,并关闭刻度显示.
for ax in axes.flat:
    add_box(ax)
    ax.axis('off')

# 第一个子图中不画出colorbar.
im = axes[0, 0].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
axes[0, 0].set_title('without colorbar')

# 第二个子图中画出依附于ax的垂直的colorbar.
im = axes[0, 1].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cbar = fig.colorbar(im, ax=axes[0, 1], orientation='vertical')
axes[0, 1].set_title('add vertical colorbar to ax')

# 第三个子图中画出依附于ax的水平的colorbar.
im = axes[1, 0].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cbar = fig.colorbar(im, ax=axes[1, 0], orientation='horizontal')
axes[1, 0].set_title('add horizontal colorbar to ax')

# 第三个子图中将垂直的colorbar画在cax上.
im = axes[1, 1].pcolormesh(X, Y, Z, cmap=cmap, shading='nearest')
cax = add_right_cax(axes[1, 1], pad=0.02, width=0.02)
cbar = fig.colorbar(im, cax=cax)
axes[1, 1].set_title('add vertical colorbar to cax')

plt.show()
```

![colorbar_and_ax](/matplotlib_colorbar/colorbar_and_axes.png)

组图通过 `plt.subplots` 函数创建，这里用红色虚线方框圈出每个子图开始时的范围。然后第一个子图内画图但不添加 colorbar，可以看到其范围与红框重合；第二个子图内用 `ax` 参数指定 colorbar 依附于该子图，可以看到子图的水平范围被 colorbar 偷走了一部分，同理第三个子图的垂直范围被偷走了一部分；而第四个子图中因为手动在子图右边创建了一个新的 axes 并指定为 `cax`，所以 colorbar 并没有挤占子图原有的空间。

总之，向主图添加 colorbar 时，`ax` 参数用起来更方便，但会改变主图的范围；`cax` 参数需要提前为 colorbar 准备一个 axes，但 colorbar 的摆放位置更为灵活。

## 2. 设置刻度

第 1 节中提到过，在初始化 colorbar 时通过 `ticks` 和 `format` 参数即可设置刻度。实际上，colorbar 在接受刻度的设置后，会将它们传给底层的 axes 对象，利用 axes 的方法来实现刻度的标注。所以为 colorbar 设置刻度有两种思路

- 利用 colorbar 提供的接口设置刻度，优点是简单直接，缺点是对于小刻度等参数无法进行细致的设定。
- 直接操作 colorbar 底层的 axes，优点是设置更细致，缺点是可能会受 `cbar.update_ticks` 方法的干扰。

正因为这两种思路都行得通，所以你上网搜如何设置刻度时能找到五花八门的方法，下面便来一一辨析这些方法。

另外需要提前说明一下，colorbar 不同于普通的 axes，只会显示落入 `cbar.vmin` 和 `cbar.vmax` 这两个值范围内的 ticks，而这两个值由 colorbar 含有的 normalization 的信息决定（例外会在后面提到）。

### 2.1 ticks 和 format 参数

```python
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=100)
im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
locator = mpl.ticker.MultipleLocator(10)
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')

cbar = fig.colorbar(
    im, cax=ax, orientation='horizontal',
    ticks=locator, format=formatter
)
cbar.minorticks_on()
```

![colorbar_ticks](/matplotlib_colorbar/colorbar_ticks.png)

直接在初始化 colorbar 的时候给出指定 `ticks` 和 `format` 参数即可。

小刻度则通过 `minorticks_on` 方法开启，可惜这个方法不提供任可控调节的参数，查看源码会发现，colorbar 是借助 `matplotlib.ticker.AutoMinorLocator` 实现小刻度的，其中小刻度的间隔数 `n` 被硬编码为默认值 `None`，所以小刻度的数目会根据大刻度的数值设为 3 个或 4 个，例如图中两个大刻度间就是 4 个小刻度。

### 2.2 locator 和 formatter 属性

```python
cbar = fig.colorbar(im, cax=ax, orientation='horizontal')
cbar.locator = locator
cbar.formatter = formatter
cbar.minorticks_on()
cbar.update_ticks()
```

图跟 2.1 节的一样。直接修改 `locator` 和 `formatter` 属性，接着调用 `update_ticks` 方法刷新刻度，将这两个属性传给底层的 axes，从而使刻度生效。2.1 节中不需要刷新是因为初始化的最后会自动刷新。

### 2.3 set_ticks 和 set_ticklabels 方法

```python
ticks = np.linspace(0, 100, 11)
ticklabels = [formatter(tick) for tick in ticks]
cbar = fig.colorbar(im, cax=ax, orientation='horizontal')
cbar.set_ticks(ticks)
cbar.set_ticklabels(ticklabels)
cbar.minorticks_on()
```

图跟 2.1 节的一样。这个方法适用于手动给出 ticks 和与之匹配的 ticklabels 的情况。同时 `set_ticks` 和 `set_ticklabels` 都有一个布尔类型的 `update_ticks` 参数，效果同 2.2 节所述，因为默认为 True，所以可以不用管它。奇怪的是，`set_ticks` 方法还可以接受 `Locator` 对象，不过当 `Locator` 与 ticklabels 对不上时就会发出警告并产生错误的结果。

也许你会联想到 axes 设置刻度的方法，并进行这样的尝试

```python
cbar.ax.set_xticks(ticks)
cbar.ax.set_xticklabels(ticklabels)
```

可惜这种方法行不通，也是会报警加出错。

### 2.4 set_major_locator 和 set_major_formatter 方法

```python
cbar = fig.colorbar(im, cax=ax, orientation='horizontal')
cbar.ax.xaxis.set_major_locator(locator)
cbar.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(2))
cbar.ax.xaxis.set_major_formatter(formatter)
# cbar.update_ticks()
```

图跟 2.1 节的一样。虽然 2.3 中直接调用 `set_xticks` 和 `set_xticklabels` 的方法失败了，但神秘的是直接调用 `set_major_locator` 和 `set_major_formatter` 却可以，你甚至可以用 `set_minor_locator` 来实现更细致的小刻度。这里因为 colorbar 是水平放置的，所以操作的是 xaxis，垂直方向换成 yaxis 即可。

这种方法的缺点是，colorbar 的 `locator` 属性与 xaxis 的并不一致

```python
In : print(cbar.locator)
Out: <matplotlib.colorbar._ColorbarAutoLocator object at 0x000001B424E36AF0>
In : print(cbar.ax.xaxis.get_major_locator())
Out: <matplotlib.ticker.MultipleLocator object at 0x000001B424E366A0>
```

尽管画出来的图是 `MultipleLocator` 的效果，但 `cbar.locator` 依旧保留初始化时的默认值，`cbar.formatter` 同理。如果此时执行 `cbar.update_ticks()`，就会将 `cbar.ax.xaxis` 的 locator 和 formatter 更新成 `cbar.locator` 和 `cbar.formatter` 的值——即变回默认效果。奇怪的是 minor locator 并不受 `update_ticks` 的影响，小刻度依然得到保留。

### 2.5 对数刻度

1.1 节中展示过，当传入的 `mappable` 的 `norm` 是 `LogNorm` 时，colorbar 会自动采取对数刻度和科学计数法的标签，并开启小刻度。下面是一个不用科学计数法，并关掉小刻度的例子

```python
norm = mpl.colors.LogNorm(vmin=1E0, vmax=1E3)
im = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

cbar = fig.colorbar(
    im, cax=ax, orientation='horizontal',
    format=mpl.ticker.ScalarFormatter()
)
cbar.minorticks_off()
```

![colorbar_log](/matplotlib_colorbar/colorbar_log.png)

### 2.6 更多设置

如果想进一步设置刻度的参数（刻度长度、标签字体等），需要通过底层的 `cbar.ax.tick_params` 方法来实现。例如

```python
cbar.ax.tick_params(length=2, labelsize='x-small')
```

总结一下的话，colorbar 提供了设置刻度的接口，但做得还不够完善，以至于我们需要直接操作底层的 axes。希望以后 Matplotlib 能对此加以改善。

## 3. Contourf 中的 colorbar

把 `pcolor`、`imshow` 等函数的返回值传给 colorbar 时，colorbar 中会显示连续完整的 colormap；但若把 `contourf` 函数的返回值传给 colorbar 时，显示的就不再是完整的 colormap，而是等高线之间的填色（填色规则请见 [Matplotlib 系列：colormap 的设置](https://zhajiman.github.io/post/matplotlib_colormap/) 第 3.1 节），下面是一个 `pcolormesh` 与 `contourf` 相对比的例子

```python
X, Y, Z = test_data()
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=100)
levels = [10, 20, 40, 80]

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax in axes:
    ax.axis('off')

# 第一张图画pcolormesh.
im = axes[0].pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
cbar = fig.colorbar(im, ax=axes[0], extend='both')
axes[0].set_title('pcolormesh')

# 第二张图画contourf.
im = axes[1].contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, extend='both')
cbar = fig.colorbar(im, ax=axes[1])
axes[1].set_title('contourf')

plt.show()
```

![pcolor_contourf](/matplotlib_colorbar/pcolor_contourf.png)

可以看到效果与上面描述的一致，colorbar 上颜色间的分界位置也与 `levels` 的数值大小相对应。第 2 节中提到过，colorbar 的显示范围由 `cbar.vmin` 和 `cbar.vmax` 决定，且这两个值与 `cbar.norm.vmin` 和 `cbar.norm.vmax` 相同——不过使用 `contourf` 的返回值作为 `mappable` 时则是例外，这里 `cbar.vmin` 和 `cbar.vmax` 由 `levels` 的边界决定。所以上图中 colorbar 的范围为 [10, 80]。

另外若 `contourf` 中指定过 `extend` 参数，那么其返回值会带有 `extend` 的信息，初始化 colorbar 时就不应该再设定 `extend` 参数了。Matplotlib 3.3 以后同时使用 `extend` 参数的行为被废弃。

## 4. 为组图添加 colorbar

### 4.1 为每个子图添加

最简单的方法是在绘制每个子图的 colorbar 时，将 `ax` 参数指定为子图的 axes，缺点是会改变子图形状，不过可以之后用 `ax.set_aspect` 等方法进行调整。下面利用 1.2 节中的 `add_right_cax` 函数实现 `cax` 的版本

```python
X, Y, Z = test_data()
cmap = mpl.cm.viridis
norm = mpl.colors.Normalize(vmin=0, vmax=100)

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
# 调节子图间的宽度,以留出放colorbar的空间.
fig.subplots_adjust(wspace=0.4)

for ax in axes.flat:
    ax.axis('off')
    cax = add_right_cax(ax, pad=0.01, width=0.02)
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(im, cax=cax)

plt.show()
```

![subplots_1](/matplotlib_colorbar/subplots_1.png)

更高级的方法是使用 `mpl_toolkits.axes_grid1.ImageGrid` 类，例如

```python
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(figsize=(8, 8))
grid = ImageGrid(
    fig, 111, nrows_ncols=(2, 2), axes_pad=0.5,
    cbar_mode='each', cbar_location='right', cbar_pad=0.1
)
# 这里ax是mpl_toolkits.axes_grid1.mpl_axes.Axes
for ax in grid:
    ax.axis('off')
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
    # 官网例子中的cax.colorbar(im)用法自Matplotlib 3.2起废弃.
    cbar = fig.colorbar(im, cax=ax.cax)

plt.show()
```

结果跟上面一张图差不多。`ImageGrid` 适合创建子图宽高比固定的组图（例如 `imshow` 的图像或等经纬度投影的地图），并且对于 colorbar 位置和间距的设置非常便利。此外还有利用 `matplotlib.gridspec.GridSpec` 和 `mpl_toolkits.axes_grid1.axes_divider` 的方法，这里就不细讲了。

### 4.2 为整个组图添加

其实 colorbar 的 `ax` 参数还可以接受 axes 组成的列表（数组），从而实现为列表中的所有 axes 只添加一个 colorbar。例如

```python
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

for ax in axes.flat:
    ax.axis('off')
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')

cbar = fig.colorbar(im, ax=axes)

plt.show()
```

![subplots_2](/matplotlib_colorbar/subplots_2.png)

再举个 `ImageGrid` 的例子

```python
fig = plt.figure(figsize=(8, 8))
grid = ImageGrid(
    fig, 111, nrows_ncols=(2, 2), axes_pad=0.5,
    cbar_mode='single', cbar_location='right', cbar_pad=0.2,
)
for ax in grid:
    ax.axis('off')
    im = ax.pcolormesh(X, Y, Z, cmap=cmap, norm=norm, shading='nearest')
    cbar = fig.colorbar(im, cax=ax.cax)

plt.show()
```

结果同上一张图。如果有更复杂的需求，例如在不改变子图形状的前提下，组图中不同区域的子图共用不同的 colorbar，那么建议使用 `add_axes` 的方法（参考 1.2 节的 `add_right_cax` 函数），或利用 `matplotlib.gridspec.GridSpec` 将 cax 穿插在组图间。感兴趣的读者可以读读参考链接中最后那篇。

## 5. 参考链接

官方教程

[Customized Colorbars Tutorial](https://matplotlib.org/stable/tutorials/colors/colorbar_only.html)

[Overview of axes_grid1 toolkit](https://matplotlib.org/stable/tutorials/toolkits/axes_grid.html)

Cartopy 的例子

[Using Cartopy and AxesGrid toolkit](https://scitools.org.uk/cartopy/docs/latest/gallery/miscellanea/axes_grid_basic.html)

可能是全网最详细的 colorbar 调整教程

[matplotlibのcolorbarを解剖してわかったこと、あるいはもうcolorbar調整に苦労したくない人に捧げる話](https://qiita.com/skotaro/items/01d66a8c9902a766a2c0)