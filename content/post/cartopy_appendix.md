---
title: "Cartopy 系列：对入门教程的补充"
date: 2021-11-06
showToc: true
tags:
- cartopy
- matplotlib
---

## 前言

几年前曾写过 [Cartopy 系列：从入门到放弃](https://zhajiman.github.io/post/cartopy_introduction/)，但现在来看还是遗漏了不少细节，比如初学者可能会遇到以下问题

- 经度是用 [-180°, 180°] 还是 [0°, 360°] 范围？
- 为什么有时候设置的刻度显示不全？
- 怎么截取跨越地图边界的区域，画图又怎么跨越边界？

本文将会用坐标变换的思想来解答以上问题，希望能给读者一些实用的启示。本来应该把这些内容写到入门教程里的，但可能会太长，所以现在单独成篇。文中的讨论主要针对最常用的 Plate Carrée 投影，其它投影需要读者自己测试。代码基于 Cartopy 0.18.0，虽然现在已经更新到 0.20.0 了，但基本思想是一致的。

<!--more-->

## 经度的循环性

经度的数值范围一般有两种表示：[-180°, 180°] 或 [0°, 360°]。前者表示以本初子午线（zero meridian）为中心，向西向东各 180°，再在对向子午线（antimeridian）处交汇；后者表示以本初子午线为起点向东 360°，又绕回了本初子午线。经度这种绕圈转的量很容易让人联想到时钟的表盘，本初子午线就对应于 0 时（实际上“子午”一词指的就是夜半和正午），[-180°, 180°] 范围对应于使用 AM 和 PM 标记的计时方式，[0°, 360°] 范围对应于二十四小时制。如下图所描绘的那样

![clock](/cartopy_appendix/clock.png)

一个小区别是：表盘的指针是顺时针旋转的，而经度的“指针”从北极往下看的话，是自西向东，也就是逆时针旋转的。

两个范围的经度在 [0°, 180°] 区间是等价的，大于 180° 的经度减去 360° 又可以换算到 [-180°, 0°] 范围内，例如 240° 就等价于 240° - 360° = -120°。在 Python 中可以通过下面的公式将 [0°, 360°] 范围的经度换算到 [-180°, 180°] 上

```python
def convert_lon(lon):
    '''将经度换算到[-180, 180]范围内.'''
    return (lon + 180) % 360 - 180

for lon in range(-270, 450 + 90, 90):
    lon_new = convert(lon)
    print(lon, '->', lon_new)
```

结果为

```
-270 -> 90
-180 -> -180
-90 -> -90
0 -> 0
90 -> 90
180 -> -180
270 -> -90
360 -> 0
450 -> 90
```

有趣的是，当经度超出了 [0°, 360°] 范围时上式依旧成立，例如 450° 表示从子午线出发绕地球一圈后再绕 90°，上面的结果中也恰好换算为 90°，同理带入 -240° 后换算成 120°。注意边界值 180° 被换算成了 -180°，不过考虑到这两个值对应于同一条经线，也还可以接受。所以只要借助这个公式，任意数值的经度都可以换算到 [-180°, 180°] 的范围内。

Cartopy 正好遵循这一特性，会自动换算我们给出的任意经度值（不过具体实现可能不同于 `convert_lon` 函数）。例如

```python
line_proj = ccrs.PlateCarree()
ax.plot([-60, 60], [0, 0], transform=line_proj)
ax.plot([300, 420], [0, 0], transform=line_proj)
```

两句 `ax.plot` 的画出来的效果是相同的，都画的是 [-60°, 60°] 之间的连线。但这并不意味着在 Cartopy 里经度只要换算过来合理，就可以随便设置了。例如对画图函数来说经度的大小顺序非常重要、对刻度设置来说因为存在 bug，效果也可能不同于预期。后面的小节会一一解说这些例外。

## 理解坐标变换

### 地理坐标与投影坐标

地理坐标即经纬度，能够描述地球表面任意一点的位置；而投影坐标则是将地球球体投影到平面上得到的坐标。二者的数值和单位一般不同，但可以根据投影时用到的数学公式进行换算。画图用的源数据（站点位置、卫星像元网格、再分析网格等）一般基于地理坐标，而 Cartopy 地图（即 GeoAxes）因为处于屏幕这个平面上，自然是基于投影坐标的。

Cartopy 将坐标系称为“坐标参考系统”（coordinate reference system，CRS），并在 `cartopy.crs` 模块中定义了一系列表示 CRS 的类，其中也包括各种地图投影，比如 `PlateCarree`、`Mercator`、`Mollweide`、`LambertConformal` 类等。在创建 Axes 时将 CRS 对象传给 `projection` 参数，即可将 Axes 转为这个 CRS 代表的投影的 GeoAxes。例如下面这段代码分别创建了等经纬度投影和麦卡托投影的地图

```python
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

proj1 = ccrs.PlateCarree()
proj2 = ccrs.Mercator()
fig = plt.figure()
ax1 = fig.add_subplot(211, projection=proj1)
ax2 = fig.add_subplot(212, projection=proj2)
```

下面以最常用的 `PlateCarree` 类为例讲解地理坐标和投影坐标的关系。`PlateCarree` 类有一个初始化参数 `central_longitude`，能够指定画全球地图（通过 `ax.set_global` 方法）时正中间的经度，默认值为 0，即全球地图默认会把本初子午线放在画面中心。若指定 `central_longitude=180`，则全球地图会以对向子午线为中心，图这里就不放了。除这个功能以外，`central_longitude` 还会影响到 `PlateCarree` 坐标与地理坐标间的关系。`PlateCarree` 是一个标准的笛卡尔坐标系，其横坐标 x 与经度 lon 满足关系

```python
x = convert_lon(lon - central_longitude)
```

即经度减去 `central_longitude` 后再换算到 [-180°, 180°] 范围即可，显然 x 可以视作关于中央经度的相对经度。继续沿用上一节的表盘比喻，将二者的关系形象地表示为下图

![rotate](/cartopy_appendix/rotate.png)

图中黑色表盘为经度 lon，将其逆时针旋转 `central_longitude` 度后即得到代表 x 的蓝色表盘。`PlateCarree` 的纵坐标 y 则与纬度 lat 直接对应，纬度是多少纵坐标就是多少。很容易注意到，当 `central_longitude=0` 时，横坐标与经度直接对应，纵坐标与经度直接对应，即 `PlateCarree` 坐标正好等价于地理坐标。我们后面还会频繁用到这一点。

举个例子，对投影 `proj=ccrs.PlateCarree(central_longitude=180)` 来说，地理坐标 `(-160, 30)` 对应于投影坐标 `(20, 30)`。这可以通过 Matplotlib 的 `plt.show` 函数创建的交互式界面得到直观验证

![plt_show](/cartopy_appendix/plt_show.png)

Matplotlib 里若把鼠标指针放在 Axes 的图像上，窗口右上角就会显示指针位置的坐标。Cartopy 的 GeoAxes 增强了这一功能，还会在坐标后面的括号里显示对应的地理坐标。如上图所示，投影坐标 `(20.32, 30.05)` 对应的地理坐标为 `(159.677419, 30.048387)`。注意图中是纬度在前经度在后，且两种坐标对小数部分的显示有所不同，所以看起来像是有误差。探索一番还能发现，全球地图里 x 的范围为 [-180°, 180°]，y 的范围为 [-90°, 90°]，地图中央，也就是 `central_longitude` 所在位置的 x 总为 0°。Matplotlib 的这一功能对日常 debug 来说非常实用。

此外 CRS 对象的 `transform_points` 方法能直接进行不同坐标系统间的坐标换算。例如

```python
import numpy as np

proj1 = ccrs.PlateCarree(central_longitude=0)
proj2 = ccrs.PlateCarree(central_longitude=180)

npt = 5
lon1 = np.linspace(-180, 180, npt)
lat1 = np.linspace(-90, 90, npt)

pos2 = proj2.transform_points(proj1, lon1, lat1)
lon2 = pos2[:, 0]
lat2 = pos2[:, 1]
for i in range(npt):
    print(f'({lon1[i]}, {lat1[i]})', '->', f'({lon2[i]}, {lat2[i]})')
```

其中 `proj1` 的中央经度为 0，如前所述，其投影坐标 `lon1` 和 `lat1` 正好代表经纬度。利用 `proj1.transform_points` 方法即可将 `lon1` 和 `lat1` 换算为 `proj2` 里的坐标 `lon2` 和 `lat2`。结果为

```
(-180.0, -90.0) -> (0.0, -90.0)
(-90.0, -45.0) -> (90.0, -45.0)
(0.0, 0.0) -> (-180.0, 0.0)
(90.0, 45.0) -> (-90.0, 45.0)
(180.0, 90.0) -> (0.0, 90.0)
```

明显 `lon2` 相当于 `lon1` 减去了 180°，而 `lat2` 和 `lat1` 完全一致。在需要手动变换坐标的场合这个方法会派上用场。

总结一下：`PlateCarree` 投影将地球投影到了平面笛卡尔坐标系里，横坐标相当于经度向右位移（逆时针旋转）了 `central_longitude` 度，纵坐标依然对应于纬度。`PlateCarree` 坐标与地理坐标的关系非常简单，但如果对于兰伯特、UTM 那种复杂的投影，坐标间的关系就不会这么直观了，甚至 x 和 y 的单位都不会是度，读者可以用前面提到的 Matplotlib 的交互式界面自行探索。

### crs 和 transform 参数

由上一节的解说，Cartopy 官方文档里着重强调的 `crs` 和 `transform` 参数就很好理解了。

GeoAxes 不仅工作在投影坐标系，其设置刻度的 `set_xticks` 和 `set_yticks` 方法、截取区域的 `set_extent` 方法，乃至各种绘图的 `plot`、`contourf`、`pcolormesh` 等方法等，都默认我们给出的数据也是基于投影坐标系的。所以需要提前把数据的地理坐标换算为地图的投影坐标，再把数据添加到地图上。例如下面这段代码

```python
map_proj = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure()
ax = fig.add_subplot(111, projection=map_proj)
ax.set_xticks([0, 90])
```

`set_xticks` 方法会在地图 `x=0` 和 `x=90` 的位置画出刻度——注意是 x 而不是经度！如果我们需要的是 `lon=0` 和 `lon=90` 处的刻度，就需要手动换算一下（根据上一节 x 和 lon 的关系式）

```python
ax.set_xticks([-180, -90])
```

`PlateCarree` 这样简单的投影还比较容易手动换算，如果是更复杂的兰伯特投影之类的，就需要利用 CRS 对象的 `transform_points` 方法了。但 Cartopy 能够通过 `crs` 和 `transform` 参数省略掉这一换算过程：通过将 CRS 对象传给设置刻度时的 `crs` 参数，或绘制图像时的 `transform` 参数，能够告知 Cartopy 你的数据基于这个 CRS 坐标系，之后 Cartopy 在内部会根据这一信息将你的数据换算到 GeoAxes 所处的坐标系中。因为我们的数据一般都基于地理坐标，所以我们常把等价于地理坐标系的 `ccrs.PlateCarree()` 对象传给 `crs` 和 `transform` 参数。例如上面在 `lon=0` 处和 `lon=90` 处标出刻度的写法可以改为

```python
tick_proj = ccrs.PlateCarree()
ax.set_xticks([0, 90], crs=tick_proj)
```

类似地，画出地理坐标 `(0, 30)` 和 `(90, 30)` 间的连线

```py
line_proj = ccrs.PlateCarree()
ax.plot([0, 90], [30, 30], transform=line_proj)
```

所以只要用好 `crs` 参数和 `transform` 参数，就可以忽略坐标转换的细节，统一使用地理坐标来描述和操作地图了。可能有人会指出，当地图投影 `map_proj=ccrs.PlateCarree()` 时 `crs` 和 `transform` 参数都可以省去，这确实没错，不过正如 Python 之禅说的，“显式胜于隐式”，显式地指定这些参数有助于明确坐标间的关系。

### Geodetic 坐标

前面说 `ccrs.PlateCarree()` 等价于地理坐标系是不严谨的，因为真正的地理坐标系定义在球面上，两点间的最短连线（测地线）是过这两点的大圆的劣弧；而 `PlateCarree` 坐标系定义在平面上，两点间的最短连线是过两点的直线。`cartopy.crs` 模块里的 `Geodetic` 类便能表示真正的地理坐标系，用于指定单点位置时其效果与 `PlateCarree` 无异，但在画两点间连线时将 `Geodetic` 对象传给 `transform` 参数，便能让连线变成球面上的测地线。例如

```python
x = [116, 286]
y = [39, 40]
ax.plot(x, y, 'o-', transform=ccrs.PlateCarree(), label='PlateCarree')
ax.plot(x, y, 'o-', transform=ccrs.Geodetic(), label='Geodetic')
ax.legend()
```

![geodetic](/cartopy_appendix/geodetic.png)

虽然乍一看橙线比蓝线长，但投影回球面后，橙线才是两点间的最短连线。`Geodetic` 是一种 CRS，但不属于地图投影，所以不能用于 GeoAxes 的创建。平时画图时除非对测地线或大圆有需求，一般使用 `PlateCarree` 坐标即可，实际上，目前 `Geodetic` 对象还不能用作 `contourf`、`pcolormesh` 等画图函数的 `transform` 参数，可能是 Matplotlib 还无法实现曲线网格的填色吧。

## 关于刻度设置

### LongitudeFormatter 和 LatitudeFormatter

单纯使用 `set_xticks` 设置刻度后，刻度会以 x 的值作为刻度标签（ticklabel），而 x 的值很可能与经度不相等。这时就需要使用 Cartopy 提供的经纬度专用的 Formatter，将刻度标签表现为正确的地理坐标的形式。例如

```python
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# 分别指定GeoAxes所处的投影和刻度所处的投影.
map_proj = ccrs.PlateCarree(central_longitude=180)
tick_proj = ccrs.PlateCarree(central_longitude=0)

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(6, 8),
    subplot_kw={'projection': map_proj}
)

# 两个ax设置相同的刻度.
for ax in axes:
    ax.set_global()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_xticks(np.linspace(-180, 180, 7), crs=tick_proj)
    ax.set_yticks(np.linspace(-90, 90, 5), crs=tick_proj)

axes[0].set_title('Ticks Added')
axes[1].set_title('Formatter Added')

# 为第二个ax使用Formatter.
axes[1].xaxis.set_major_formatter(LongitudeFormatter())
axes[1].yaxis.set_major_formatter(LatitudeFormatter())

plt.show()
```

![ticks_and_formatter](/cartopy_appendix/ticks_and_formatter.png)

可以看到上图中的刻度标签显示的是 x 的值，下图中 Formatter 通过读取 GeoAxes 的投影信息，将刻度值换算为经纬度，并追加了度数和方向的符号。`LongitudeFormatter` 和 `LatitudeFormatter` 还提供丰富的参数来修改刻度的显示效果，不过一般来说默认设置就够用了。另外这两个 Formatter 还可以用于普通的 Axes，会将 Axes 的坐标视为地理坐标。

### set_xticks 和 gridlines 的 bug

`set_xticks` 方法存在 bug：当省略 `crs` 参数，或提供的 CRS 对象与 GeoAxes 的投影等价（源码里通过 `==` 判断）时，会跳过坐标变换的环节，直接使用你提供的刻度。例如

```python
map_proj = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(111, projection=map_proj)
ax.set_global()

ax.set_xticks(np.linspace(0, 360, 7), crs=map_proj)
ax.set_yticks(np.linspace(-90, 90, 5), crs=map_proj)
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())
```

![ticks_bug_1](/cartopy_appendix/ticks_bug_1.png)

本来 `set_xticks` 里大于 180° 的刻度需要先换算到 [-180°, 180°] 范围内，现在这一环节被跳过了，大于 180° 的刻度直接标在了地图外面。弥补方法是，刻度改用 `np.linspace(-180, 180, 7)` 即可，或者当 `crs` 参数与 `map_proj` 不同时，错误也会自动消失。

画网格的 `gridlines` 方法存在类似的问题：超出 [-180°, 180°] 范围的经度刻度直接画不出来，就算 `crs` 参数不同于 `map_proj` 也没用。例如

```python
map_proj = ccrs.PlateCarree(central_longitude=180)
tick_proj = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(111, projection=map_proj)
ax.set_global()

ax.gridlines(
    crs=tick_proj, draw_labels=True,
    xlocs=np.linspace(0, 360, 7),
    ylocs=np.linspace(-90, 90, 5),
    color='k', linestyle='--'
)
```

![ticks_bug_2](/cartopy_appendix/ticks_bug_2.png)

可以看到西半球的经度网格线没画出来，并且调用 `fig.savefig` 保存图片时若 `dpi` 不为默认的 150，连纬度的标签也会莫名其妙消失（另见 [issues 1794](https://github.com/SciTools/cartopy/issues/1794)）。Bug 具体原因我也不清楚，感兴趣的读者可以自己探究一下。弥补方法是一样的，`xlocs` 改用 `np.linspace(-180, 180, 7)` 即可。

## 跨越边界的 plot

本节探讨通过 `plot` 方法绘制两点间连线时，在什么情况下会跨越边界相连。测试程序如下

```python
map_proj = ccrs.PlateCarree()
tick_proj = ccrs.PlateCarree()
fig, axes = plt.subplots(
    nrows=2, ncols=2, figsize=(10, 6),
    subplot_kw={'projection': map_proj}
)
fig.subplots_adjust(wspace=0.3)

# 填色和设置刻度.
for ax in axes.flat:
    ax.set_global()
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.set_xticks(np.linspace(-180, 180, 7), crs=tick_proj)
    ax.set_yticks(np.linspace(-90, 90, 5), crs=tick_proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

def draw_line(ax, p1, p2):
    '''画出点p1和p2之间的连线,并标注在标题上.'''
    x0, y0 = p1
    x1, y1 = p2
    line_proj = ccrs.PlateCarree()
    ax.plot([x0, x1], [y0, y1], 'o-', c='C3', transform=line_proj)
    ax.text(
        x0, y0 + 15, 'start', ha='center', va='center',
        transform=line_proj
    )
    ax.text(
        x1, y1 + 15, 'end', ha='center', va='center',
        transform=line_proj
    )
    ax.set_title(f'From {p1} to {p2}')

draw_line(axes[0, 0], (120, 60), (240, -60))
draw_line(axes[0, 1], (240, -60), (120, 60))
draw_line(axes[1, 0], (120, 60), (-120, -60))
draw_line(axes[1, 1], (-120, -60), (120, 60))

plt.show()
```

![plot](/cartopy_appendix/plot.png)

从测试结果可以归纳出：设起点的坐标为 `(x0, y0)`，终点的坐标为 `(x1, y1)`，接着比较 `x0` 和 `x1` 的绝对大小，当 `x0 < x1` 时，会从起点出发自西向东绘制；当 `x0 > x1` 时，会从起点出发自东向西绘制。例如左上角的图中，起点的经度数值小于终点，所以向东绘制，且中途穿越了地图边界；右上角的图将起点和终点颠倒后，变为从起点出发向西绘制；左下角和右下角的图同理，但不穿越地图边界。借助这一特性，我们可以预测并控制两点间的连线是走“内圈”（不穿越边界），还是走“外圈”（穿越边界）。

这点不仅限于 `plot` 方法，`contourf`、`pcolormesh`、`imshow` 等其它绘图方法，乃至截取区域用的 `set_extent` 方法均遵循这一特性。

## 跨越边界的 set_extent

上一节提到 `set_extent` 方法会根据 `x0` 和 `x1` 的大小关系决定绕圈方向，但实际上想要成功截取还需要范围不能跨过边界。例如

```python
clon1 = 0
clon2 = 180
map_proj1 = ccrs.PlateCarree(central_longitude=clon1)
map_proj2 = ccrs.PlateCarree(central_longitude=clon2)
data_proj = ccrs.PlateCarree()
extent = [120, 240, 20, 80]
lonmin, lonmax, latmin, latmax = extent

# 第一行和第二行子图的central_longitude不同.
fig = plt.figure(figsize=(12, 8))
ax1 = fig.add_subplot(221, projection=map_proj1)
ax2 = fig.add_subplot(222, projection=map_proj1)
ax3 = fig.add_subplot(223, projection=map_proj2)
ax4 = fig.add_subplot(224, projection=map_proj2)
fig.subplots_adjust(hspace=-0.1)

for ax in [ax1, ax3]:
    ax.set_global()
    ax.set_xticks(np.linspace(-180, 180, 7), crs=data_proj)
    ax.set_yticks(np.linspace(-90, 90, 5), crs=data_proj)
    # 用patch标出extent范围.
    patch = mpatch.Rectangle(
        (lonmin, latmin), lonmax - lonmin, latmax - latmin,
        fc='C3', alpha=0.4, transform=data_proj
    )
    ax.add_patch(patch)

for ax in [ax2, ax4]:
    ax.set_xticks(np.linspace(lonmin, lonmax, 7), crs=data_proj)
    ax.set_yticks(np.linspace(latmin, latmax, 4), crs=data_proj)
    # 截取区域
    ax.set_extent(extent, crs=data_proj)

# 填色和添加formatter.
for ax in [ax1, ax2, ax3, ax4]:
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())

# 设置标题.
ax1.set_title(f'central_longitude={clon1}°')
ax3.set_title(f'central_longitude={clon2}°')
ax2.set_title('set_extent failed')
ax4.set_title('set_extent success')

plt.show()
```

![set_extent](/cartopy_appendix/set_extent.png)

截取范围为经度 [120°, 240°]，纬度 [20°, 80°]。第一排图片 `central_longitude=0`，红色方块标识出了截取范围，可以看到这张图中截取范围跨越了地图边界（180°），然后右边对纬度的截取成功了，但对经度的截取失败了——经度范围仍然是 [-180°, 180°]，所以地图变成了长条状。第二排图片 `central_longitude=180`，此时地图边界变为 0°，截取范围因此没有跨越边界，然后右边得到了正确的截取结果。

由此引出了 `central_longitude` 的又一作用：控制地图边界，以保证 `set_extent` 生效。额外再提一点，使用 `set_extent` 截取完后，若再调用 `set_xticks` 和 `set_yticks` 画超出截取范围的刻度时，会强制拓宽当前地图的范围。所以建议先设置刻度，再进行截取（这点对 `set_global` 也是一样的）。 

## GeoAxes 的位置

Matplotlib 中可以通过 `set_aspect` 方法调整 Axes 横纵坐标单位的比例，例如 `ax.set_aspect(1)` 使横纵坐标单位比例为 1:1，因而图片上一个单位的 x 和一个单位的 y 代表的物理长度（英寸）相等。

之所以要提这一点，是因为 GeoAxes（即地图）的横纵坐标单位比例必须保持不变，不然随便调整一下 `figsize`、`rect`、`xlim`、`ylim` 等参数，地图就会变形，相当于地图的投影被改变了。例如等经纬度投影的地图单位经度和单位纬度必须等长，否则就名不副其实了。用 `PlateCarree` 对象创建 GeoAxes 时，Cartopy 会自动进行类似于 `ax.set_aspect(1)` 的操作，以满足这一条件。

不过由此也会带来一个问题：你无法在改变比例的同时维持 Axes 的形状不变（特指 `adjustable='box'` 时，详见 [matplotlib.axes.Axes.set_adjustable](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_adjustable.html)）。例如用 `fig.add_axes` 创建 `PlateCarree` 投影的 GeoAxes 时，可以用 `rect` 参数指定 GeoAxes 方框的大小和位置，但如前所述，GeoAxes 会自动设置比例为 1，所以最后画出来的地图方框很可能并不符合 `rect`。下面用代码进行演示

```python
from matplotlib.transforms import Bbox

proj = ccrs.PlateCarree()
fig = plt.figure()
rect = [0.2, 0.2, 0.6, 0.6]
axpos1 = Bbox.from_bounds(*rect)
ax = fig.add_axes(rect, projection=proj)
ax.set_global()
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.OCEAN)
axpos2 = ax.get_position()

# 画出rect的方框.
patch = mpatch.Rectangle(
    (axpos1.x0, axpos1.y0), axpos1.width, axpos1.height,
    ec='C3', fc='none', transform=fig.transFigure
)
fig.patches.append(patch)
fig.text(
    axpos1.x0, axpos1.y0 + axpos1.height, 'Expected Box',
    c='C3', va='bottom'
)

# 画出地图的方框.
patch = mpatch.Rectangle(
    (axpos2.x0, axpos2.y0), axpos2.width, axpos2.height,
    ec='C0', fc='none', transform=fig.transFigure
)
fig.patches.append(patch)
fig.text(
    axpos2.x0 + axpos2.width, axpos2.y0 + axpos2.height,
    'Actual Box', c='C0', ha='right', va='bottom'
)

print('Expected Box:', axpos1)
print('Actual Box:', axpos2)
plt.show()
```

打印结果为

```
Expected Box: Bbox(x0=0.2, y0=0.2, x1=0.8, y1=0.8)
Actual Box: Bbox(x0=0.2, y0=0.30000000000000004, x1=0.8, y1=0.7000000000000002)
```

![box_1](/cartopy_appendix/box_1.png)

可以看到地图的实际方框维持中心位置和宽度不变，但对恒定比例的要求使其高度缩短了。实际上，若通过 `set_extent` 方法截取区域，还可能出现实际方框高度不变、宽度缩短的情况，这里就不放图片了。总之是想说明，`PlateCarree` 投影的 GeoAxes 常常出现会出现高度或宽度短于预期的情况。其实际大小位置可以通过 `get_position` 方法获取，之后可以用于绘制等高或等宽的 colorbar 等（例子可见 [Matplotlib 系列：colorbar 的设置](https://zhajiman.github.io/post/matplotlib_colorbar/)）。

强行把地图填到 `rect` 指示的空间里也不是不行，只需要设置

```python
ax.set_aspect('auto')
```

![box_2](/cartopy_appendix/box_2.png)

不过这样一来投影就称不上等经纬度投影了。

## 结语

文中很多经验都是笔者试出来的，Cartopy 的官方文档并没有详细解说，所以这些经验可能存在不严谨或错误的地方，还请读者在评论区指出。

## 参考链接

[Cartopy API reference](https://scitools.org.uk/cartopy/docs/latest/reference/index.html)

[Longitude conversion 0~360 to -180~180](https://confluence.ecmwf.int/display/CUSF/Longitude+conversion+0~360+to+-180~180)

[preventing spurious horizontal lines for ungridded pcolor(mesh) data](https://stackoverflow.com/questions/46527456/preventing-spurious-horizontal-lines-for-ungridded-pcolormesh-data)

[Force aspect ratio for a map](https://stackoverflow.com/questions/15480113/force-aspect-ratio-for-a-map)
