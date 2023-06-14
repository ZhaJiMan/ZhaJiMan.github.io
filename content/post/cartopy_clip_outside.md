---
title: "Cartopy 系列：裁剪填色图出界问题"
date: 2023-06-14
showToc: true
tags:
- cartopy
- matplotlib
---

## 前言

裁剪或者说白化，就是让填色图只显示在多边形里面，不显示在多边形外面，例如只显示 `GeoAxes.contourf` 在中国境内的结果。实现方法为：

```Python
from matplotlib.path import Path
from cartopy.mpl.patch import geos_to_path
from cartopy.io.shapereader import Reader

reader = Reader(filepath)
geom = next(reader.geometries())
reader.close()

cf = ax.contourf(X, Y, Z, transform=crs)
geom = ax.projection.project_geometry(geom, crs)
path = Path.make_compound_path(*geos_to_path(geom))
for col in cf.collections:
    col.set_clip_path(path, ax.transData)
```

- 将 `crs` 坐标系上的多边形对象变换到 data 坐标系上。
- 利用 `geos_to_path` 和 `make_compound_path` 将变换后的多边形转为 `Path` 对象。
- 对 `QuadContourSet.collections` 里的每个成员调用 `set_clip_path` 方法，并且指定 data 坐标系。

![fig1](/cartopy_clip_outside/fig1.png)

<!--more-->

完整代码为：

```Python
import numpy as np
import shapefile
import shapely.geometry as sgeom
from shapely.ops import unary_union
import matplotlib.pyplot as plt
from matplotlib.path import Path
import cartopy.crs as ccrs
from cartopy.mpl.patch import geos_to_path

def test_data():
    '''生成测试的二维数据.'''
    x = np.linspace(70, 140, 100)
    y = np.linspace(10, 60, 100)
    X, Y = np.meshgrid(x, y)
    Z = X + Y
    
    return X, Y, Z

def load_country():
    '''读取中国国界线数据.'''
    filepath = './data/bou2_4p.shp'
    with shapefile.Reader(filepath, encoding='gbk') as reader:
        provinces = list(map(sgeom.shape, reader.shapes()))
    country = unary_union(provinces)
    
    return country

def make_map(extents):
    '''创建地图.'''
    map_crs = ccrs.LambertConformal(
        central_longitude=105,
        standard_parallels=(25, 47)
    )
    data_crs = ccrs.PlateCarree()

    fig = plt.figure()
    ax = fig.add_subplot(projection=map_crs)
    ax.set_extent(extents, crs=data_crs)
    ax.coastlines()

    return ax

X, Y, Z = test_data()
country = load_country()
crs = ccrs.PlateCarree()

ax = make_map([75, 135, 10, 60])
ax.add_geometries(country, crs, fc='none', ec='k')
cf = ax.contourf(X, Y, Z, levels=20, transform=crs)

geom = ax.projection.project_geometry(country, crs)
path = Path.make_compound_path(*geos_to_path(geom))
for col in cf.collections:
    col.set_clip_path(path, ax.transData)
```

但当地图的显示范围比用来裁剪的形状要小时，就会出现填色图溢出地图边界的情况。下面以东南区域为例：

![fig2](/cartopy_clip_outside/fig2.png)

创建矩形边界小地图的代码为：

```Python
ax = make_map([100, 125, 15, 40])
```

创建扇形边界小地图的代码为：

```Python
ax = make_map([100, 125, 15, 40])
verts = [(100, 15), (125, 15), (125, 40), (100, 40), (100, 15)]
rect = Path(verts).interpolated(100)
ax.set_boundary(rect, crs)
```

发现填色图虽然被国界裁剪了，但西部和东北区域溢出了地图的边界，这个效果显然是不可接受的。本文的目的是解释其原因并给出两种通用且简单的解决方法。文中 Catopy 版本为 0.21。

## 出界的原因

`Artist.clipbox` 属性是一个矩形的边界框，能够在绘制 `Artist` 时不让它超出这个框框的范围。`Artist._clippath` 属性是 `Path` 对象，能够在绘制 `Artist` 时裁剪它。`Path` 对象可以是任意形状，可以是带洞的多边形，可以由多个多边形组成，只要在构造 `Path` 时设定好 `codes` 参数即可。刚创建的 `Artist` 的这两个属性都为 `None`，表示不做裁剪；`Artist` 被添加到 `Axes` 上时，会用代表显示范围的矩形的 `Axes.patch` 属性作为 `clipbox`。因此 `Axes.plot` 和 `Axes.contourf` 等方法画出来的结果从来都不会有出界的情况。

一般 `Artist._clippath` 属性始终为 `None`，我们可以通过 `Artist.set_clip_path` 方法来设定它，并且注意到其优先级低于 `_clipbox`。所以如果你在普通的 `Axes` 上做过地图裁剪的话，会发现并没有填色图出界的问题。实际上，出界是因为 `GeoAxes.patch` 并不一定是矩形的，例如全球范围的 Lambert 投影地图的边界是展开的圆锥，Mollweide 投影地图的边界是一个椭圆。为了让 `Artist` 的内容不超出形状各异的边界，Cartopy 选择将 `GeoAxes.patch` 赋给 `_clippath`，`clipbox` 保持为 `None`（即便地图边界实际上是矩形）。

简言之，Cartopy 在画图时已经用地图的边界裁剪了填色图，我们之后再用中国国界做裁剪，就会破坏掉原来的裁剪效果。当中国国界小于地图边界时不会露陷，而大于时就会出现填色图超出地图边界的问题。

## 解决方法

### 设定 bbox

注意到 `Axes` 和 `GeoAxes` 都有 `bbox` 属性，也能表示轴的边界框。当地图边界是矩形时，`GeoAxes.patch` 和 `GeoAxes.bbox` 表示相同的范围，因此设定 `Artist.clipbox` 来裁去出界的部分：

```Python
for col in cf.collections:
    col.set_clip_path(path, ax.transData)
    col.set_clip_box(ax.bbox)
```

![fig3](/cartopy_clip_outside/fig3.png)


只用加一行，矩形边界地图的出界问题就解决了。但扇形边界的地图里，左上角仍有少许出界的部分。因为 `GeoAxes.bbox` 只是框住整个 `GeoAxes` 的方框，而 `GeoAxes.patch` 不一定与之重合。为此下面再给出第二种方法。

### 与地图边界求与

思路是提取地图边界在 data 坐标系里的坐标点，构造一个多边形对象，与做过坐标变换的、同样在 data 坐标系里的国界多边形求与（即取两个多边形相重叠的部分），用得到的新多边形去做裁剪。代码为：

```Python
patch = ax.patch
ax.draw_artist(patch)
trans = patch.get_transform() - ax.transData
path = patch.get_path().transformed(trans)
boundary = sgeom.Polygon(path.vertices)

geom = ax.projection.project_geometry(country, crs)
geom = geom & boundary
path = Path.make_compound_path(*geos_to_path(geom))
for col in cf.collections:
    col.set_clip_path(path, ax.transData)
```

`GeoAxes.patch` 一般基于 data 坐标系，但如果调用过 `GeoAxes.set_boundary`，也可能变到其它坐标系上，因此这里通过 `Transform` 对象的减法操作来得到 data 坐标系上的坐标点。同时注意到，`GeoAxes.patch` 的具体数值是在渲染过程中决定的，所以需要先调用 `Axes.draw_artist` 或 `Canvas.draw` 方法。效果如下图：

![fig4](/cartopy_clip_outside/fig4.png)

Bbox 法代码简单，但是不能正确处理非矩形边界的地图，并且有些情况下耗时更长；求与法能保证效果，但如果之后修改地图的显示范围，或者在交互模式中进行拖拽，则会出现填色图缺漏的情况。

## 结语

本文找出了 Cartopy 裁剪填色图出界的原因，并给出了两种解决方法。但两种方法都不算完美，也许应该考虑在 `draw_event` 事件中进行裁剪并缓存 `Path` 对象？如果读者有好的方法的话还请多多交流。

另外笔者上传的 frykit 包里实现了求与法，可以通过 `clip_by_cn_border` 函数直接用国界裁剪 `contourf` 和 `pcolormesh` 等画图结果，感兴趣的读者也可以用用。

## 参考链接

[matplotlib.transforms](https://matplotlib.org/stable/api/transformations.html)

[Apply set_clip_path to contours, but the set_extend is not work. #1580](https://github.com/SciTools/cartopy/issues/1580)

[contour.set_clip_path(clip) beyond borders #2052](https://github.com/SciTools/cartopy/issues/2052)