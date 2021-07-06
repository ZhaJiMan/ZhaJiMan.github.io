---
title: "Cartopy 系列：为 Lambert 投影地图添加刻度"
date: 2021-03-24
showToc: true
tags:
- cartopy
---
## 前言

Cartopy 中的 Plate Carrée 投影使用方便，但在展示中国地图时会使中国的形状显得很瘪，与之相比，Lambert 投影的效果会更加美观，下图显示了两种投影的差异

![comparison](/cartopy_lambert/comparison.png)

所以本文将会介绍如何在 Cartopy 中实现 Lambert 投影，并为地图添上合适的刻度。文中 Cartopy 的版本是 0.18.0。

<!--more-->

## Lambert 投影的简单介绍

这里的 Lambert 投影指的是 Lambert conformal conic 投影（兰勃特等角圆锥投影），是通过让圆锥面与地球相切（割），然后将地球表面投影到圆锥面上来实现的。作为一种等角地图投影，Lambert 投影能够较好地保留区域的角度和形状，适合用于对中纬度东西方向分布的大陆板块进行制图。详细的描述请见维基和 [ArcMap 上的介绍](https://desktop.arcgis.com/zh-cn/arcmap/latest/map/projections/lambert-conformal-conic.htm)。

在 Cartopy 中，这一投影通过 `LambertConformal` 类来实现

```Python
import cartopy.crs as ccrs

map_proj = ccrs.LambertConformal(
    central_longitude=105, standard_parallels=(25, 47)
)
```

这个类的参数有很多，这里为了画出中国地图，只需要指定中央经线 `central_longitude=105`，两条标准纬线 `standard_parallels=(25, 47)`，参数来源是 [中国区域Lambert&Albers投影参数](http://blog.sina.com.cn/s/blog_4aa4593d0102ziux.html) 这篇博文。其实笔者对这些参数也没什么概念，如果有错误还请读者指出。

按照这个设置便可以画出全球的地图了，并且中国位于地图中心

![global](/cartopy_lambert/global.png)

## 用 set_extent 方法截取区域

我们一般需要通过 `GeoAxes` 的 `set_extent` 方法截取我们关心的区域，下面截取 80°E-130°E，15°N-55°N 的范围

```Python
extent = [80, 130, 15, 55]
ax.set_extent(extent, crs=ccrs.PlateCarree())
```

结果如下图，原本扇形的全球地图会被截取成矩形

![set_extent](/cartopy_lambert/set_extent.png)

道理上来说给出经纬度的边界，截取出来的应该是一个更小的扇形，但按 [issue #697](https://github.com/SciTools/cartopy/issues/697) 的说法，`set_extent` 会选出一个刚好包住这个小扇形的矩形作为边界。这里来验证一下这个说法

```Python
import matplotlib as mpl
rect = mpl.path.Path([
    [extent[0], extent[2]],
    [extent[0], extent[3]],
    [extent[1], extent[3]],
    [extent[1], extent[2]],
    [extent[0], extent[2]]
]).interpolated(20)
line = rect.vertices
ax.plot(line[:, 0], line[:, 1], lw=1, c='r', transform=ccrs.Geodetic())
```

这段代码是将 `extent` 所描述的小扇形画在地图上，结果在上一张图里有。可以看到，这个小扇形确实刚好被矩形边界给包住。

如果确实需要截取出扇形的区域，可以用 `set_boundary` 方法，效果如下图

```Python
ax.set_boundary(rect, transform=ccrs.Geodetic())
```

![set_boundary](/cartopy_lambert/set_boundary.png)

截取后反而中国显示不全了，需要重新调整 `extent` 的值。

## 为地图添加刻度——默认方法

Cartopy 的版本在 0.17 及以下时，只支持给 Plate Carrée 和 Mercator 投影的地图添加刻度。一个变通的方法是用 `ax.text` 方法手动添加刻度标签，例子见 [Python气象绘图教程](http://bbs.06climate.com/forum.php?mod=viewthread&tid=95948) 的第 18 期。

等到了最新的 0.18 版本，`gridlines` 方法有了给**所有投影**添加刻度标签的能力。下面来测试一下

```Python
ax.gridlines(
    xlocs=np.arange(-180, 180 + 1, 10), ylocs=np.arange(-90, 90 + 1, 10),
    draw_labels=True, x_inline=False, y_inline=False,
    linewidth=0.5, linestyle='--', color='gray'
)
```

`xlocs` 与 `ylocs` 指定网格线的经纬度位置，实际上超出地图边界的网格并不会被画出，所以这里给出的范围比较宽。`draw_labels` 指示是否画出刻度标签，而 `x_inline` 与 `y_inline` 指示这些标签是否画在地图里面。inline 的选项开启后效果比较乱，所以这里都关闭。结果如下图

![default_1](/cartopy_lambert/default_1.png)

默认的效果十分拉胯，四个方向上都有标签，并且有着多余的旋转效果。那么再修改 `gl`的属性 看看

```Python
# 关闭顶部和右边的标签,同时禁用旋转.
gl.top_labels = False
gl.right_labels = False
gl.rotate_labels = False
```

![](/cartopy_lambert/default_2.png)

结果改善了很多，但仍然有很奇怪的地方：虽然关闭了右边的纬度标签，但经度的标签出现在了两边的 y 轴上。根据 [issue #1530](https://github.com/SciTools/cartopy/issues/1530)，一个很不优雅的解决方法是将网格线分两次来画

- 第一次画出纬线和 90°E-120°E 的经线，并且 `draw_label=True`。

- 第二次单独画出 70°E、80°E、130°E、140°E 的经线，并且 `draw_label=False`。

结果这里就不展示了，肯定能去掉 y 轴上的经度标签，但显然这个方法有点“事后擦屁股”的意思。

## 为地图添加刻度——自制方法

这里尝试自己写一个添加刻度的函数。思路来自 Cartopy 的 `Gridliner` 类的源码和 [
Labelling grid lines on a Lambert Conformal projection](https://gist.github.com/ajdawson/dd536f786741e987ae4e) 这篇 note。

原理是想办法在 Lambert 投影坐标系（这里亦即 Matplotlib 的 data 坐标系）下表示出 xy 轴和网格线的空间位置，若一条网格线与一个轴线相交，那么这个交点的位置即刻度的位置。最后直接将这些位置用于 `set_xticks` 和 `set_yticks` 方法。判断两线相交用到了 Shapley 库。代码和效果如下

```Python
import numpy as np
import shapely.geometry as sgeom

import matplotlib.pyplot as plt

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def find_x_intersections(ax, xticks):
    '''找出xticks对应的经线与下x轴的交点在data坐标下的位置和对应的ticklabel.'''
    # 获取地图的矩形边界和最大的经纬度范围.
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    xaxis = sgeom.LineString([(x0, y0), (x1, y0)])
    # 仅选取能落入地图范围内的ticks.
    lon_ticks = [tick for tick in xticks if tick >= lon0 and tick <= lon1]

    # 每条经线有nstep个点.
    nstep = 50
    xlocs = []
    xticklabels = []
    for tick in lon_ticks:
        lon_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.full(nstep, tick),
                np.linspace(lat0, lat1, nstep)
            )[:, :2]
        )
        # 如果经线与x轴有交点,获取其位置.
        if xaxis.intersects(lon_line):
            point = xaxis.intersection(lon_line)
            xlocs.append(point.x)
            xticklabels.append(tick)
        else:
            continue

    # 用formatter添上度数和东西标识.
    formatter = LongitudeFormatter()
    xticklabels = [formatter(label) for label in xticklabels]

    return xlocs, xticklabels

def find_y_intersections(ax, yticks):
    '''找出yticks对应的纬线与左y轴的交点在data坐标下的位置和对应的ticklabel.'''
    x0, x1, y0, y1 = ax.get_extent()
    lon0, lon1, lat0, lat1 = ax.get_extent(ccrs.PlateCarree())
    yaxis = sgeom.LineString([(x0, y0), (x0, y1)])
    lat_ticks = [tick for tick in yticks if tick >= lat0 and tick <= lat1]

    nstep = 50
    ylocs = []
    yticklabels = []
    for tick in lat_ticks:
        # 注意这里与find_x_intersections的不同.
        lat_line = sgeom.LineString(
            ax.projection.transform_points(
                ccrs.Geodetic(),
                np.linspace(lon0, lon1, nstep),
                np.full(nstep, tick)
            )[:, :2]
        )
        if yaxis.intersects(lat_line):
            point = yaxis.intersection(lat_line)
            ylocs.append(point.y)
            yticklabels.append(tick)
        else:
            continue

    formatter = LatitudeFormatter()
    yticklabels = [formatter(label) for label in yticklabels]

    return ylocs, yticklabels

def set_lambert_ticks(ax, xticks, yticks):
    '''
    给一个LambertConformal投影的GeoAxes在下x轴与左y轴上添加ticks.

    要求地图边界是矩形的,即ax需要提前被set_extent方法截取成矩形.
    否则可能会出现错误.

    Parameters
    ----------
    ax : GeoAxes
        投影为LambertConformal的Axes.

    xticks : list of floats
        x轴上tick的位置.

    yticks : list of floats
        y轴上tick的位置.

    Returns
    -------
    None
    '''
    # 设置x轴.
    xlocs, xticklabels = find_x_intersections(ax, xticks)
    ax.set_xticks(xlocs)
    ax.set_xticklabels(xticklabels)
    # 设置y轴.
    ylocs, yticklabels = find_y_intersections(ax, yticks)
    ax.set_yticks(ylocs)
    ax.set_yticklabels(yticklabels)
```

![custom](/cartopy_lambert/custom.png)

这次的效果就好很多了，并且相比于默认方法，坐标轴上也有了刻度的凸起。需要注意的是，这个方法要求在设置刻度之前就通过 `set_extent` 方法截取出矩形的边界，否则可能有奇怪的结果。另外经测试对 Albers 投影也适用。

也许下次更新后 Cartopy 的刻度标注功能能得到改善，就算没有，我们也可以根据上面描述的思路来自制刻度。
