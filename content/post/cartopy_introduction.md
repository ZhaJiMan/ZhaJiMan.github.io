---
title: "Cartopy 系列：从入门到放弃"
date: 2021-03-23
showToc: true
tags:
- cartopy
- matplotlib
---

## 简介

常用的地图可视化的编程工具有 MATLAB、IDL、GrADS、GMT、NCL 等。我之前一直使用的是脚本语言 NCL，易用性不错，画地图的效果也很好。然而 2019 年初，NCAR 宣布 NCL 将停止更新，并会在日后转为 Python 的绘图包。于是我开始考虑转投 Python，同时觉得在 Python 环境下如果还是用 PyNGL 那一套语法的话，未免有些换汤不换药。因此我选择用 Python 环境下专有的 Cartopy 包来画地图。

![cartopy_log](/cartopy_introduction/cartopy_log.png)

此前 Python 最常用的地图包是 Basemap，然而它将于 2020 年被弃用，官方推荐使用 Cartopy 包作为替代。Cartopy 是英国气象局开发的地图绘图包，实现了 Basemap 的大部分功能，还可以通过 Matplotlib 的 API 实现丰富的自定义效果。

本文将会从一个 NCL 转 Python 的入门者的角度，介绍如何安装 Cartopy，如何绘制地图，并实现一些常用的效果。代码基于 0.18.0 版本的 Cartopy。

<!--more-->

## 提示

本文其实更新过数次，每次都修正了一些表述或 bug，如果还存在问题的话请读者在评论区指出。另外建议读完本文后继续阅读 [Cartopy 系列：对入门教程的补充](https://zhajiman.github.io/post/cartopy_appendix/)，解答了更多常见的问题。也可以了解一下笔者开发的画中国地图的包 [frykit](https://zhajiman.github.io/post/frykit/)。

## 安装 Cartopy 和相关的库

通过 Conda 来安装 Cartopy 是最为简单方便的。首先我们需要下载最新的 Python 3 的 Conda 环境（Anaconda 或 Miniconda 皆可），设置国内镜像源，建立好虚拟环境，然后参照 Cartopy 官网的 [installation guide](https://scitools.org.uk/cartopy/docs/latest/installing.html)，执行操作：

```
conda install -c conda-forge cartopy
```

接着便会开始安装 Cartopy，以及 Numpy、Matplotlib 等一系列相关包。Cartopy 的安装就是这么简单。之后还可以考虑去安装 netCDF4、h5py、pyhdf 等支持特定数据格式读写的包。

## 画地图的基本流程

以一个简单的例子来说明：

```Python
# 导入所需的库
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# 创建画布以及ax
fig = plt.figure()
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())

# 调用ax的方法画海岸线
ax.coastlines()

plt.show()
```

![coastlines](/cartopy_introduction/coastlines.png)

Cartopy 是利用 Matplotlib 来画图的，因此首先要导入 `pyplot` 模块。在 Cartopy 中，每种投影都是一个类，被存放在 `cartopy.crs` 模块中，crs 即坐标参考系统（Coordinate Reference Systems）之意。所以接着要导入这个模块。这里选取最常用的等距圆柱投影 `ccrs.PlateCarree` 作为地图投影。

我们知道，Matplotlib 画图是通过调用 `Axes` 类的方法来完成的。Cartopy 创造了一个 `Axes` 的子类，`GeoAxes`，它继承了前者的基本功能，还添加了一系列绘制地图元素的方法。创建一个 `GeoAxes` 对象的办法是，在创建 axes（或 subplot）时，通过参数 `projection` 指定一个 `ccrs` 中的投影。这里便利用这一方法生成了一个等距圆柱投影下的 ax。

最后调用 ax 的方法 `coastlines` 画出海岸线，默认以本初子午线为中心，比例尺为 1:110m（m 表示 million）。

因此用 Cartopy 画地图的基本流程并不复杂：

- 创建画布。
- 通过指定 `projection` 参数，创建 `GeoAxes` 对象。
- 调用 `GeoAxes` 的方法画图。

## GeoAxes 的一些有用的方法

`GeoAxes` 有不少有用的方法，这里列举如下：

- `set_global`：让地图的显示范围扩展至投影的最大范围。例如，对 `PlateCarree` 投影的 ax 使用后，地图会变成全球的。
- `set_extent`：给出元组 `(x0, x1, y0, y1)` 以限制地图的显示范围。
- `set_xticks`：设置 x 轴的刻度。
- `set_yticks`：设置 y 轴的刻度。
- `gridlines`：给地图添加网格线。
- `coastlines`：在地图上绘制海岸线。
- `stock_img`：给地图添加低分辨率的地形图背景。
- `add_feature`：给地图添加特征（例如陆地或海洋的填充、河流等）。

后文中具体的例子中将会经常用到这些方法。

## 使用不同的投影

```Python
# 选取多种投影
projections = [
    ccrs.PlateCarree(),
    ccrs.Robinson(),
    ccrs.Mercator(),
    ccrs.Orthographic()
]

# 画出多子图
fig = plt.figure()
for i, proj in enumerate(projections, 1):
    ax = fig.add_subplot(2, 2, i, projection=proj)
    ax.stock_img()  # 添加低分辨率的地形图
    ax.coastlines()
    ax.set_title(f'{type(proj)}', fontsize='small')

plt.show()
```

![projections](/cartopy_introduction/projections.png)

这个例子展示了如何使用其它投影和画出多子图。其中 `stock_img` 方法可以给地图添加低分辨率的地形背景图，让地图显得不那么寒碜。

在初始化投影时可以指定一些参数，例如 `ccrs.PlateCarree(central_longitude=180)` 可以让等距圆柱投影的全球图像的中央位于太平洋的 180 度经线处。

画多子图还可以用 `plt.subplots` 函数，但是投影就只能通过 `subplot_kw` 参数给出，并且每张子图的投影要求一致。

## 在地图上添加特征（Features）

除了画出海岸线外，我们常常需要在地图上画出更多特征，例如陆地海洋、河流湖泊等。`cartopy.feature` 中便准备了许多常用的特征对象。需要注意的是，这些对象的默认比例是 1:110m。

![features_web](/cartopy_introduction/features_web.png)
```Python
import cartopy.feature as cfeature

fig = plt.figure()
proj = ccrs.PlateCarree()
ax = fig.add_subplot(111, projection=proj)

# 设置经纬度范围,限定为中国
# 注意指定crs关键字,否则范围不一定完全准确
extents = [75, 150, 15, 60]
ax.set_extent(extents, crs=proj)
# 添加各种特征
ax.add_feature(cfeature.OCEAN)
ax.add_feature(cfeature.LAND, edgecolor='black')
ax.add_feature(cfeature.LAKES, edgecolor='black')
ax.add_feature(cfeature.RIVERS)
ax.add_feature(cfeature.BORDERS)
# 添加网格线
ax.gridlines(linestyle='--')

plt.show()
```
![features.png](/cartopy_introduction/features.png)

`add_feature` 方法能够把 `cfeature` 里的特征对象添加到地图上。上面的例子中就依次添加了海洋、陆地、湖泊、河流，还有国界线的特征。由于渲染实际上采用的是 Matplotlib 里 [annotations](https://matplotlib.org/tutorials/text/annotations.html) 的方法，所以添加的特征本质上就是一些线或者多边形，`edgecolor`、`facecolor` 等常用关键字都可以用来指定这些特征的效果。

Cartopy 本身自带一些常用的地图数据，不过有些特征并没有内置，而是会在脚本运行时自动从 Natural Earth 网站上下载下来，此时命令行可能会提示一些警告信息。下载完成后，以后使用这个特征都不会再出现警告。

另外存在一个非常重要的问题，Cartopy自带的中国地图数据不符合我国的地图标准，例如上图中缺少台湾地区，藏南区域边界有误。后面的小节还会再提到如何画出正确的中国地图。

## 设置地图分辨率

![natural_earth](/cartopy_introduction/natural_earth.png)

Cartopy 自带的 Natural Earth 的地图有三档分辨率：1:10m、1:50m、1:110m。默认分辨率为 1:110m，这在很多场合下显得很粗糙。设置分辨率的方法如下：

```Python
# coastlines方法使用resolution关键字
ax.coastlines(resolution='50m')
# add_feature方法中,则要调用cfeature对象的with_scale方法
ax.add_feature(cfeature.OCEAN.with_scale('50m'))
```

接着是一个例子：

```Python
fig = plt.figure()
res = ['110m', '50m', '10m']
extents = [75, 150, 15, 60]

proj = ccrs.PlateCarree()
for i, res in enumerate(['110m', '50m', '10m']):
    ax = fig.add_subplot(1, 3, i+1, projection=proj)
    ax.set_extent(extents, crs=proj)

    ax.add_feature(cfeature.OCEAN.with_scale(res))
    ax.add_feature(cfeature.LAND.with_scale(res), edgecolor='black')
    ax.add_feature(cfeature.LAKES.with_scale(res), edgecolor='black')
    ax.add_feature(cfeature.RIVERS.with_scale(res))
    ax.add_feature(cfeature.BORDERS.with_scale(res))
    ax.gridlines(linestyle='--')

    ax.set_title('resolution=' + res)

plt.show()
```

![resolutions](/cartopy_introduction/resolutions.png)

可以看到绘制效果有很大区别，不过相应地，分辨率越高画图速度越慢。

## 下载地图

Cartopy 自带的地图数据保存在下面这个命令显示的目录中

```python
import cartopy
print(cartopy.config['data_dir'])
```

一般来说自带的地图足以满足日常需求，如果想手动下载地图，可以到 [Natural Earth](https://www.naturalearthdata.com/) 网站上下载所需的地图数据。该网页提供三类地图数据：

- Cultural：国界线、道路、铁路等文化信息。
- Physical：陆地、海洋、海岸线、湖泊、冰川等地质信息。
- Raster：各种分辨率的地形起伏栅格文件。

其中 Cultural 和 Physical 数据可以作为常用的特征来进行添加，而 Raster 数据则需要用 `imshow` 方法来作为图片显示。把下载好的文件解压到 `data_dir` 下对应的子目录中即可。

## 在地图上添加数据

在直接调用 `ax.plot`、`ax.contourf` 等方法在地图上添加数据之前，需要了解 Cartopy 的一个核心概念：在创建一个 `GeoAxes` 对象时，通过 `projection` 关键字指定了这个地图所处的投影坐标系，这个坐标系的投影方式和原点位置都可以被指定。但是我们手上的数据很可能并不是定义在这个坐标系下的（例如那些规整的经纬度网格数据），因此在调用画图方法往地图上添加数据时，需要通过 `transform` 关键字指定我们的数据所处的坐标系。画图过程中，Cartopy 会自动进行这两个坐标系之间的换算，把我们的数据正确投影到地图的坐标系上。下面给出一个例子：

```Python
# 定义一个在PlateCarree投影中的方框
x = [-100.0, -100.0, 100.0, 100.0, -100.0]
y = [-60.0, 60.0, 60.0, -60.0, -60.0]

# 选取两种地图投影
map_proj = [ccrs.PlateCarree(), ccrs.Mollweide()]
data_proj = ccrs.PlateCarree()

fig = plt.figure()
ax1 = fig.add_subplot(211, projection=map_proj[0])
ax1.stock_img()
ax1.plot(x, y, marker='o', transform=data_proj)
ax1.fill(x, y, color='coral', transform=data_proj, alpha=0.4)
ax1.set_title('PlateCarree')

ax2 = fig.add_subplot(212, projection=map_proj[1])
ax2.stock_img()
ax2.plot(x, y, marker='o', transform=data_proj)
ax2.fill(x, y, color='coral', transform=data_proj, alpha=0.4)
ax2.set_title('Mollweide')

plt.show()
```

![add_data](/cartopy_introduction/add_data.png)

可以看到，等距圆柱投影地图上的一个方框，在摩尔威投影的地图上会向两边“长胖”——尽管这两个形状代表同一个几何体。如果不给出 `transform` 关键字，那么 Cartopy 会默认数据所在的坐标系是 `PlateCarree()`。为了严谨起见，建议在使用任何画图方法（`plot`、`contourf`、`pcolormesh` 等）时都给出 `transform` 关键字。

## 为地图添加经纬度刻度

在 0.17 及以前的版本中，**Cartopy 仅支持为直角坐标系统（等距圆柱投影和麦卡托投影）添加刻度**，而对兰勃特投影这样的则无能为力。0.18 版本开始，虽然官网说已经实现了对所有投影添加刻度的功能（[PR #1117](https://github.com/SciTools/cartopy/pull/1117)），但实际效果还是挺奇怪。因此这里就只以等距圆柱投影为例

```Python
# 导入Cartopy专门提供的经纬度的Formatter
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

map_proj = ccrs.PlateCarree()
fig = plt.figure()
ax = fig.add_subplot(111, projection=map_proj)

ax.set_global()
ax.stock_img()

# 设置大刻度和小刻度
tick_proj = ccrs.PlateCarree()
ax.set_xticks(np.arange(-180, 180 + 60, 60), crs=tick_proj)
ax.set_xticks(np.arange(-180, 180 + 30, 30), minor=True, crs=tick_proj)
ax.set_yticks(np.arange(-90, 90 + 30, 30), crs=tick_proj)
ax.set_yticks(np.arange(-90, 90 + 15, 15), minor=True, crs=tick_proj)

# 利用Formatter格式化刻度标签
ax.xaxis.set_major_formatter(LongitudeFormatter())
ax.yaxis.set_major_formatter(LatitudeFormatter())

plt.show()
```

![set_ticks](/cartopy_introduction/set_ticks.png)

Cartopy 中需要用 `GeoAxes` 类的 `set_xticks` 和 `set_yticks` 方法来分别设置经度和纬度刻度。这两个方法还可以通过 `minor` 参数，指定是否添上小刻度。其中 `crs` 关键字指的是我们给出的刻度是在什么坐标系统下定义的，这样好换算至 ax 所在的坐标系统，原理同上一节所述。如果不指定，就很容易出现把刻度画到地图外的情况。除了 `set_xticks`，`set_extent` 方法同样有 `crs` 关键字，我们需要多加注意。

接着利用 Cartopy 专门提供的 Formatter：`LongitudeFormatter` 和 `LatitudeFormatter` 来格式化刻度标签，使之能有东经西经、南纬北纬的字母标识。值得一提的是，这两个类还能用在普通的 `Axes` 上（例如拿来画纬高图）。

在标识刻度的过程中，有时可能会出现下图这样的问题

![tick_error](/cartopy_introduction/tick_error.png)

即全球地图的最右端缺失了 0° 的标识，这是 Cartopy 内部在换算刻度时用到了 mod 计算而导致的，解决方法见 stack overflow 上的 [这个讨论](https://stackoverflow.com/questions/56412206/cant-show-0-tick-in-right-when-central-longitude-180)，这里就不赘述了。额外提一句，NCL 对于这种情况就能正确处理。

Cartopy 还有一个很坑的地方在于，`set_extent` 与指定刻度的效果会互相覆盖：如果先用前者设置好了地图的显示范围，接下来的 `set_xticks` 超出了 `extents` 的范围的话，最后的出图范围就会以刻度的范围为准。因此使用时要注意刻度的范围，或把 `set_extent` 操作放在最后实施。

除了利用 `set_xticks` 和 `set_yticks` 方法，还可以在画网格线的同时画出刻度。例子如下：

```Python
ax = plt.axes(projection=ccrs.Mercator())
ax.coastlines()

# 开启网格线
gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True,
    linewidth=1, color='gray', linestyle='--',
    xlocs=[-180, -45, 0, 45, 180],
    ylocs=np.arange(-80, 81, 20)
)
# 隐藏上边和左边的刻度标签
gl.top_labels = False
gl.left_labels = False
# 设置刻度标签的风格
gl.xlabel_style = {'color': 'red', 'weight': 'bold'}
gl.ylabel_style = {'size': 10, 'color': 'gray'}

plt.show()
```

![gridline](/cartopy_introduction/gridline.png)

`gridlines` 方法可以为地图添加网格线，其中 `xlocs` 和 `ylocs` 关键字指定经纬度刻度（还可以接受 Locator），`crs` 参数指定刻度所属的坐标系统，`xformatter` 和 `yformatter` 关键字指定刻度的 Formatter——不过默认即为 `LongitudeFormatter` 和 `LatitudeFormatter`，所以这里可以省略。这种方法的优点是网格线 `gl` 所属的 `Gridliner` 类有丰富的可调选项，缺点是这些刻度并非真正意义上的刻度，而只是网格线的标签，所以坐标轴上会缺少凸出的线条。

## 绘制正确的中国地图

我在网上找到了两个绘制中国地图的教程：

- [捍卫祖国领土从每一张地图开始](https://mp.weixin.qq.com/s/jpZOpnFvMwi4ZTafflXIzw)
- [Cartopy 绘图示例库](https://gnss.help/2018/04/24/cartopy-gallery/index.html)

第一个链接提供了正确的中国省界的 shapefile，用 Cartopy 的 shapereader 读取后即可绘制。第二个链接则利用的是 GMT 中文社区上提供的省界的经纬度数据。两个链接都给出了完整的代码，经测试都可以正常作图。第一个链接的效果图如下：

![china_map](/cartopy_introduction/china_map.png)

问题在于两种方法的画图速度都非常慢，可能是因为给出的 shapefile 分辨率太高？我自己用的是 [Meteoinfo](http://meteothink.org/) 里自带的 bou2_4p.shp 文件，这个文件分辨率适中，画图速度比较理想。使用方法同第一个链接。

## 从入门到放弃

最后来个 NCL 与 Cartopy 在画图方面的简单对比吧。

**NCL：**

- 画地图参数多，效果好，官方文档详尽。
- 画图速度较快。
- 绘图语法虽然麻烦，但能写出很规整的代码。
- 默认的画图模板不好看，改善效果很麻烦。

**Cartopy：**

- 画地图的可调参数比 NCL 少，需要通过 Matplotlib 魔改上去。
- 官方文档信息不全，缺乏例子，有问题只能靠 Stack Overflow。
- 画图速度偏慢。
- 画等经纬度投影的效果还行，但是对于其它投影经常会有 bug。
- pcolormesh 等方法绘制的图像在跨越 0° 经度时常常会出问题。
- 与 Matplotlib 配合较好。

总之，我现在觉得，除非是对 Python 丰富的扩展库有需求的话，单就画点科研用的地图，从 NCL 转 Python 并没有太大的优势，还会让你陷入同 bug 作战的漩涡中。NCL 语言虽然冷门，但它从上世纪90年代发展至今，版本号已经达到 6.6.2，多年下来已经累计了足够多的实用功能。虽然这一优秀的工具停止了开发，但它依旧适用于一般的数据处理和可视化工作。

不过技多不压身，学点 Cartopy，就当是熟悉一下 Python 的功能吧。

## 画图的例子

下面举一个读取 NETCDF 格式的 ERA5 文件并画图的例子。首先在 `map_funcs.py` 文件里定义一些常用的自定义函数

```python
#----------------------------------------------------------------------------
# 2019-09-10
# 绘制地图用的函数.
#----------------------------------------------------------------------------
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def add_Chinese_provinces(ax, **kwargs):
    '''
    在地图上画出中国省界的shapefile.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.

    **kwargs
        绘制shape时用到的参数.例如linewidth,edgecolor和facecolor等.
    '''
    proj = ccrs.PlateCarree()
    reader = shpreader.Reader('D:/maps/shps/bou2_4p.shp')
    provinces = reader.geometries()
    ax.add_geometries(provinces, proj, **kwargs)
    reader.close()

def set_map_extent_and_ticks(
    ax, extents, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None
):
    '''
    设置矩形投影的地图的经纬度范围和刻度.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.支持_RectangularProjection和Mercator投影.

    extents : 4-tuple of float or None
        经纬度范围[lonmin, lonmax, latmin, latmax].值为None表示全球.

    xticks : array_like
        经度主刻度的坐标.

    yticks : array_like
        纬度主刻度的坐标.

    nx : int, optional
        经度主刻度之间次刻度的个数.默认没有次刻度.
        当经度不是等距分布时,请不要进行设置.

    ny : int, optional
        纬度主刻度之间次刻度的个数.默认没有次刻度.
        当纬度不是等距分布时,请不要进行设置.

    xformatter : Formatter, optional
        经度主刻度的Formatter.默认使用无参数的LongitudeFormatter.

    yformatter : Formatter, optional
        纬度主刻度的Formatter.默认使用无参数的LatitudeFormatter.
    '''
    # 设置主刻度.
    proj = ccrs.PlateCarree()
    ax.set_xticks(xticks, crs=proj)
    ax.set_yticks(yticks, crs=proj)
    # 设置次刻度.
    xlocator = mticker.AutoMinorLocator(nx + 1)
    ylocator = mticker.AutoMinorLocator(ny + 1)
    ax.xaxis.set_minor_locator(xlocator)
    ax.yaxis.set_minor_locator(ylocator)

    # 设置Formatter.
    if xformatter is None:
        xformatter = LongitudeFormatter()
    if yformatter is None:
        yformatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(xformatter)
    ax.yaxis.set_major_formatter(yformatter)

    # 在最后调用set_extent,防止刻度拓宽显示范围.
    if extents is None:
        ax.set_global()
    else:
        ax.set_extent(extents, crs=proj)

def add_box_on_map(ax, extents, **rect_kw):
    '''
    在地图上画出一个方框.

    Parameters
    ----------
    ax : GeoAxes
        目标地图.最好为矩形投影,否则效果可能很糟.

    extents : 4-tuple of float
        方框的经纬度范围[lonmin, lonmax, latmin, latmax].

    **rect_kw
        创建Rectangle时的关键字参数.
        例如linewidth,edgecolor和facecolor等.
    '''
    lonmin, lonmax, latmin, latmax = extents
    rect = mpatches.Rectangle(
        (lonmin, latmin), lonmax - lonmin, latmax - latmin,
        transform=ccrs.PlateCarree(), **rect_kw
    )
    ax.add_patch(rect)
```

其中 `add_Chinese_provinces` 函数用于在地图上画出中国省界的 shapefile；`set_map_extent_and_ticks` 用于设置矩形投影（例如 `PlateCarree`）地图的显示范围和刻度，代码参考了 Cartopy 和 [GeoCAT-viz](https://github.com/NCAR/geocat-viz) 的源码。接着是主程序

```Python
#-------------------------------------------------------------------------
# 2019-09-10
# 画出ERA5数据在500hPa高度的相对湿度和水平风场.
#-------------------------------------------------------------------------
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import cartopy.crs as ccrs

from map_funcs import add_Chinese_provinces, set_map_extent_and_ticks

if __name__ == '__main__':
    # 设置绘图区域.
    lonmin, lonmax = 75, 150
    latmin, latmax = 15, 60
    extents = [lonmin, lonmax, latmin, latmax]

    # 读取extents区域内的数据.
    filename = 't_uv_rh_gp_ERA5.nc'
    with xr.open_dataset(filename) as ds:
        # ERA5文件的纬度单调递减,所以先反转过来.
        ds = ds.sortby(ds.latitude)
        ds = ds.isel(time=0).sel(
            longitude=slice(lonmin, lonmax),
            latitude=slice(latmin, latmax),
            level=500
        )

    proj = ccrs.PlateCarree()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=proj)

    # 添加海岸线和中国省界.
    ax.coastlines(resolution='10m', lw=0.3)
    add_Chinese_provinces(ax, lw=0.3, ec='k', fc='none')
    # 设置经纬度刻度.
    set_map_extent_and_ticks(
        ax, extents,
        xticks=np.arange(-180, 190, 15),
        yticks=np.arange(-90, 100, 15),
        nx=1, ny=1
    )
    ax.tick_params(labelsize='small')

    # 画出相对湿度的填色图.
    im = ax.contourf(
        ds.longitude, ds.latitude, ds.r,
        levels=np.linspace(0, 100, 11), cmap='RdYlBu_r',
        extend='both', alpha=0.8
    )
    cbar = fig.colorbar(
        im, ax=ax, shrink=0.9, pad=0.1, orientation='horizontal',
        format=mticker.PercentFormatter()
    )
    cbar.ax.tick_params(labelsize='small')

    # 画出风箭头.直接使用DataArray会报错,所以转换成ndarray.
    Q = ax.quiver(
        ds.longitude.values, ds.latitude.values,
        ds.u.values, ds.v.values,
        scale_units='inches', scale=180, angles='uv',
        units='inches', width=0.008, headwidth=4,
        regrid_shape=20, transform=proj
    )
    # 在ax右下角腾出放图例的空间.
    # zorder需大于1,以避免被之前画过的内容遮挡.
    w, h = 0.12, 0.12
    rect = mpatches.Rectangle(
        (1 - w, 0), w, h, transform=ax.transAxes,
        fc='white', ec='k', lw=0.5, zorder=1.1
    )
    ax.add_patch(rect)
    # 添加风箭头的图例.
    qk = ax.quiverkey(
        Q, X=1-w/2, Y=0.7*h, U=40,
        label=f'{40} m/s', labelpos='S', labelsep=0.05,
        fontproperties={'size': 'x-small'}
    )

    title = 'Relative Humidity and Wind at 500 hPa'
    ax.set_title(title, fontsize='medium')

    fig.savefig('rh_wnd.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
```

![example](/cartopy_introduction/example.png)

其中绘制风箭头的部分可以参考 [Matplotlib 系列：图解 quiver](https://zhajiman.github.io/post/matplotlib_quiver/)。

## 补充链接

本文介绍的只是 Cartopy 的最简单的功能，还有诸如读取 shapefile、地图 mask、使用网络地图等功能都没有介绍（因为我也没用到过……）。下面补充一些可能有帮助的链接

- 一个地球与环境数据科学的教程：[Making Maps with Cartopy](https://earth-env-data-science.github.io/intro)
- 云台书使的绘图教程，内容非常全面，含有地图裁剪等高级内容：[Python气象绘图教程](http://bbs.06climate.com/forum.php?mod=viewthread&tid=95948)
- Unidata 给出的例子：[Unidata Example Gallery](https://unidata.github.io/python-training/gallery/gallery-home/)
- GeoCAT 给出的仿 NCL 的例子：[GeoCAT-examples](https://geocat-examples.readthedocs.io/en/latest/index.html)
- Cartopy 开发成员对于数据跨越边界时的解说：[preventing spurious horizontal lines for ungridded pcolor(mesh) data](https://stackoverflow.com/questions/46527456/preventing-spurious-horizontal-lines-for-ungridded-pcolormesh-data)

- 谈到了如何重复利用底图的方法：[(筆記) python的cartopy使用、清除已畫的資料方法](https://home.gamer.com.tw/creationDetail.php?sn=4318912)