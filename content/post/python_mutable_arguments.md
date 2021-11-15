---
title: "Python 系列：小心默认的可变参数"
date: 2021-11-14
tags:
- python
---

之前我在 [Cartopy 系列：从入门到放弃](https://zhajiman.github.io/post/cartopy_introduction/) 一文中定义了这样一个函数

```python
def set_map_extent_and_ticks(
    ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=LongitudeFormatter(),
    yformatter=LatitudeFormatter()
):
    ...
```

<!--more-->

其功能是限制 GeoAxes 的经纬度范围，并画出经纬度刻度。其中 `LongitudeFormatter` 和 `LatitudeFormatter` 是 Cartopy 定义的两个 Formatter 类，用于格式化经纬度刻度标签。Formatter 对象因为其属性可以任意修改，所以也可以算作可变对象（[are user defined classes mutable](https://stackoverflow.com/questions/12076445/are-user-defined-classes-mutable)）。[What the f*ck Python!](https://github.com/leisurelicht/wtfpython-cn) 中提到过

> Python中函数的默认可变参数并不是每次调用该函数时都会被初始化。相反，它们会使用最近分配的值作为默认值，除非明确地将可变对象传递给函数。

也就是说，多次调用 `set_map_extent_and_ticks` 时如果不指定 `xformatter` 和 `yformatter`，就会一直沿用第一次调用时创建的 Formatter 对象，这一点可以通过打印对象的 id 来验证。而 Formatter 作为一种 Matplotlib Artist，被重复使用时可能会产生错误的结果（[早く知っておきたかったmatplotlibの基礎知識、あるいは見た目の調整が捗るArtistの話](https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9)）。我就因为对不同投影的多个 GeoAxes 连续使用 `set_map_extent_and_ticks`，画出了错误的刻度。

避免可变参数导致的错误的常见做法是将 `None` 指定为参数的默认值，在函数体内判断是否创建默认的可变对象。所以这个函数应该修改为

```python
def set_map_extent_and_ticks(
    ax, extent, xticks, yticks, nx=0, ny=0,
    xformatter=None, yformatter=None
):
    ...
```
