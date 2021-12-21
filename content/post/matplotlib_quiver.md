---
title: "Matplotlib 系列：图解 quiver"
date: 2021-12-18
showToc: true
tags:
- matplotlib
- cartopy
---

## 前言

Matplotlib 中用箭头表示风场或电磁场等矢量场时需要用到 `quiver` 方法，据字典，quiver 一词的意思是颤动、颤抖或箭袋，貌似也就最后一个意思跟箭头搭得上边。相比于其它画图方法，`quiver` 的参数又多又容易混淆，所以本文将以图解的方式逐一介绍。这些参数按功能可分为三种：控制箭头位置和数值的、控制箭头长度和角度的，以及控制箭头尺寸和形状的。下面会按照这个分组顺序来解说。本文代码基于 Matplotlib 3.3.4。

<!--more-->

## 箭头的位置和数值

据文档，`quiver` 的函数签名为

```python
quiver([X, Y], U, V, [C], **kw)
```

- `X` 和 `Y` 指定矢量及箭头的位置。
- `U` 和 `V` 指定矢量的横纵分量。
- `C` 数组的数值会通过 `cmap` 和 `norm` 映射为箭头的颜色（原理详见 [Matplotlib 系列：colormap 的设置](https://zhajiman.github.io/post/matplotlib_colormap/)），例如可以取矢量长度 `np.hypot(U, V)`。如果只是想让所有箭头颜色相同，使用 `color` 参数即可。

`quiver` 既可以像 `scatter` 那样接受一维散点数据，画出任意位置的箭头，也可以像 `pcolormesh` 那样绘制二维网格数据。

`pivots` 参数可以指定 `X` 和 `Y` 的位置对应于箭头的尾部、中间，还是头部，默认 `pivot = tail`，即箭头从 `X` 和 `Y` 的位置出发。下面基于这个设置讲解箭头的长度和角度。

## 箭头的长度和角度

箭头的长度和角度能直接反映矢量的强度和方向，所以控制这些量的参数无疑是最重要的。其中长度由 `scale_units` 和 `scale` 两个参数控制，角度由 `angles` 参数控制。对于一个分量为 `(u, v)` 的矢量来说，其在 uv 空间里的长度和角度分别为

```
len_vector = sqrt(u**2 + v**2)
angle_vector = arctan(v / u)
```

箭头是画在 `Axes` 的 xy 空间里的，从矢量到箭头要经过两个空间之间的变换。首先介绍如何得到箭头长度

```
len_arrow = len_vector / scale [scale_units]
```

其中 `scale` 用于放缩数值，`scale_units` 决定箭头的长度单位。所谓单位即某个基准长度，需要参考图中已有的元素来进行设定。例如当 `scale = 1` 时，箭头长度等于矢量长度的数值乘上这个基准长度。`scale_units` 可取七种：`'inches'`、`'dots'`、`'width'`、`'height'`、`'x'`、`'y'` 和 `'xy'`。下图展示了前六种

![blank](/matplotlib_quiver/blank.png)

该图由 `fig, ax = plt.subplots()` 语句生成，默认 `figsize = (6.4, 4.8)`，`dpi = 100`，所以尺寸为 6.4 x 4.8 英寸，或 640 x 480 像素（英寸和像素的意义详见 [Matplotlib 系列：导出高 DPI 的图片](https://zhajiman.github.io/post/matplotlib_dpi/)）。以 `inches` 为例，若 `scale = 1`，那么长度为 1 的矢量在图上对应于长度为 1 英寸的箭头，其它单位同理。图中未展示的 `'xy'` 单位比较特殊，后面讲到 `angles` 时再细说。

七种单位中 `inches` 和 `dots` 显然是绝对单位，而剩下的均为相对于 `Axes` 的元素设定的单位。在 `plt.show` 弹出的交互式窗口内缩放 `Axes` 时，基于相对单位的箭头长度会动态变化，而基于绝对单位的箭头长度则纹丝不动。无论选用哪种单位，若箭头过长或过短，都可以用 `scale` 参数缩放到合适的范围：`scale` 越小，箭头越长；`scale` 越大，箭头越短。

接着来看如何得到箭头角度。控制箭头角度的 `angles` 有三种设置：一是把单个浮点数或数组传给 `angles` 参数，直接指定每个箭头的角度，此时矢量的 `u` 和 `v` 分量和箭头角度没有任何关系。二是令 `angles = 'uv'`，表示沿用矢量角度

```
angle_arrow = angle_vector
```

三是令 `angles = 'xy'`，一般需要和 `scale_units = 'xy'` 联用，此时箭头等同于 xy 平面里 `(x, y)` 到 `(x + u, y + v)` 的连线箭头。例如当 xy 平面是空间位置，矢量表示位移时就适合用这个设置。下面示意 `angles` 的效果

![angles](/matplotlib_quiver/angles.png)

图中为了体现 uv 空间和 xy 空间的差异，特地设置 `ax.set_aspect(0.5)` ，于是网格单元的宽高比为 2:1。可以看到，`angles = 'uv'` 时，箭头角度就为 45°；`angles = 'xy'` 且 `scale_units = 'xy'` 时，箭头与网格单元的对角线刚好重合。这里未展示 `angles` 为定值的结果，是因为 `scale_units = 'xy'` 与之冲突，导致画不出箭头，也许是个 bug。

`scale_units` 和 `scale` 默认为 `None`，表示 Matplotlib 会自动根据矢量长度的平均值，以及矢量的个数决定箭头的长度。`angles` 默认为 `'uv'`。一般我们只需要调整 `scale_units` 和 `scale`，而不需要改动 `angles`。

值得一提的是，若通过 `ax.set_aspect(1)` 使 `Axes` 两个坐标轴的单位长度等长，那么 `'x'`、`'y'` 和 `'xy'` 三种长度单位的结果相同， `'uv'` 和 `'xy'` 两种角度设置的结果也相同。

## 箭头的尺寸和形状

类似于箭头长度与 `scale_units` 的关系，箭头尺寸的单位由 `units` 给出，同样可取七种：`'inches'`、`'dots'`、`'width'`、`'height'`、`'x'`、`'y'`、`'xy'`。此处 `'xy'` 的含义不同于上一节，仅指 `Axes` 对角线的单位长度。`units` 默认为 `width`。

选好单位后首先需要设置的参数是 `width`，箭杆（shaft）的宽度就等于 `width` 的数值乘上单位对应的基准长度。之后其它形状参数——`headwidth`、`headlength`、`headaxislength`——均以箭杆的宽度为单位。下图描绘了这些参数代表的部分

![shape](/matplotlib_quiver/shape.png)

`width` 默认为 `None`，表示 Matplotlib 会自动决定箭杆宽度。而其它参数都有提前设好的值，例如 `headwidth` 默认为 3，表示箭镞（允许我用古文称呼箭头尖尖）的宽度总是箭杆的三倍。

最后提一个神秘的地方，文档指出 `units` 不会影响箭头长度，但事实是在不给出 `scale_units` 时，`units` 会同时决定箭头长度和尺寸的单位。例如参考资料的最后一篇便展示了 `units` 对箭头长度的影响，我个人认为这是 Matplotlib 的设计失误。

## 箭头的阈值

你可能会碰到箭头的尺寸不合预期、或箭头缩成了一个点的情形，这都是 `minshaft` 和 `minlength` 这两个阈值参数导致的。

`minshaft` 以 `headlength` 为单位，默认为 1，当箭头长度小于 `minshaft` 代表的长度时，箭头整体尺寸会按箭头长度等比例缩小。

`minlength` 以 `width` 为单位，默认为 1，当箭头长度小于 `minlength` 代表的长度时，箭头直接退化成以该长度为直径的六边形。

选用默认值的场合，`minshaft` 是五倍 `width` 的长度，`minlength` 是单倍 `width` 的长度，当矢量长度越来越小时，对应的箭头一开始只缩短长度，后来尺寸也跟着缩小，最后直接缩成一个点（六边形）。如果没有这两个参数，那么特别短的矢量在图上仍然会挂着一个特别大的箭镞，既不美观，还可能影响我们的判断。下面改编一个 [官网示例](https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_simple_demo.html)

![min](/matplotlib_quiver/min.png)

可以看到左图中间的短矢量与周围的长矢量通过尺寸差异被区分开来，而右边则很难辨认，中间的箭头还出现了空心情况。这两个阈值一般不需要改动，默认条件下就有不错的效果。

## 箭头的图例

箭头的图例通过 `quiverkey` 方法添加，由一个箭头和文本标签构成。函数签名为

```
quiverkey(Q, X, Y, U, label, **kwargs)
```

下面列举常用参数：

- `Q`：`quiver` 方法返回的 `Quiver` 对象，借此可以画出与 `quiver` 类似的箭头。
- `X` 和 `Y`：图例的位置。虽然用大写字母表示，其实并不是数组。
- `U`：箭头的长度，用矢量长度衡量。
- `label`：标签的文本，一般填 `U` 的数值和矢量的单位。
- `coordinates`：指定 `X` 和 `Y` 是什么坐标，可选 `'axes'`、`'figure'`、`'data'` 和 `'inches'`，默认为 `'axes'`。坐标间的差异请见文档的 [Transformations Tutorial](https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html#sphx-glr-tutorials-advanced-transforms-tutorial-py)。
- `labelpos`：标签相对于箭头的位置，可选 `'N'`、`'S'`、`'E'` 和 `'W'`。默认为北，即标签在箭头上面。
- `labelsep`：标签与箭头间的距离，默认为 0.1 pt。
- `fontproperties`：用于指定标签字体参数的字典。

[Cartopy 系列：从入门到放弃](https://zhajiman.github.io/post/cartopy_introduction/) 文末提供了一个示例，同时为了实现 NCL 那种箭头图例外面带个方框的风格，在图例后面还加了个矩形补丁。

## Cartopy 中的 quiver

Cartopy 的 `GeoAxes` 对 `Axes` 的 `quiver` 方法进行了装饰，使之能通过 `transform` 参数实现不同 CRS 间的坐标变换（详见 [Cartopy 系列：对入门教程的补充](https://zhajiman.github.io/post/cartopy_appendix/)）。注意所有投影的 `GeoAxes` 的 `aspect_ratio` 都为 1，所以正如本文开头提到的，`scale_units` 取 `x`、`y` 或 `xy` 时结果没区别，`angles` 取 `uv` 或 `xy` 结果也没有区别。尽管如此，考虑到各种投影坐标系的 x 范围和 y 范围通常都很怪，胆小的我还是会取 `scale_units = 'inches'`，`angles = 'uv'`。

此外 Cartopy 还提供了一个非常便利的新参数 `regrid_shape`，可以将矢量场重新插值到投影坐标系中的规则网格上，以达到规整矢量位置或稀疏箭头密度的目的，而在 `Axes` 中这活儿需要通过手动插值或跳步索引来实现。`regrid_shape` 接收二元组或整数，前者指定 x 和 y 方向上的箭头个数，后者指定短边上的箭头个数，然后长边的个数通过地图范围的宽高比缩放得出。默认为 `None`，即不进行网格化。下面改编一个 [官网示例](https://scitools.org.uk/cartopy/docs/latest/gallery/vector_data/regridding_arrows.html)

![cartopy](/matplotlib_quiver/cartopy.png)

两图中的风场基于 `NorthPolarStereo` 坐标里的规则网格，地图则基于 `PlateCarree` 坐标。上图未进行网格化，风箭头明显间距不一。下图指定 `regrid_shape = 10` 后，风场被 `scipy.interpolate.griddata` 函数线性插值到地图上形为 `(16, 10)` 的规则网格中，箭头因而清晰可辨。

## 结语

文中未给出渐变色箭头的例子，读者可以参考 [官网的 demo](https://matplotlib.org/stable/gallery/images_contours_and_fields/quiver_demo.html)。另外矢量场除了用 `quiver` 画箭头表示，还可以用 `streamplot` 画流线表示，以后有机会再另行介绍。

## 参考资料

[matplotlib.axes.Axes.quiver](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiver.html)

[matplotlib.axes.Axes.quiverkey](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.quiverkey.html#matplotlib.axes.Axes.quiverkey)

[cartopy.mpl.geoaxes.GeoAxes.quiver](https://scitools.org.uk/cartopy/docs/latest/reference/generated/cartopy.mpl.geoaxes.GeoAxes.html#cartopy.mpl.geoaxes.GeoAxes.quiver)

[【python】quiverの矢印の長さをうまく調整したい【matplotlib.pyplot.quiver】](https://chemstat.hatenablog.com/entry/2020/10/26/050441)
