---
title: "通过缩放和偏移压缩数据"
date: 2024-01-20
showToc: false
tags:
- 卫星
- 翻译
---

> ERA5 的 NetCDF 文件或卫星的 HDF 文件为了压缩文件体积会用 16 位整数存储变量，读取时跟属性里的 `add_offset` 和 `scale_factor` 做运算恢复成 64 位浮点数。如果你是用 Python 的 NetCDF4 或 xarray 包处理 NetCDF 文件，甚至都不用关心这些细节，它们默认会帮你解包成浮点数。问题是，如果自己也想用这种方法压缩数据，那么 `add_offset` 和 `scale_factor` 该如何设置，压缩率能有多高，又会损失多少精度呢？一番搜索后发现 [Unidata Developer's Blog](https://www.unidata.ucar.edu/blogs/developer/en/) 上的博文 [Compression by Scaling and Offfset](https://www.unidata.ucar.edu/blogs/developer/entry/compression_by_scaling_and_offfset)（原文标题确实把 offset 拼错了）清晰地介绍了压缩的原理和参数选择，现翻译前半部分，后半部分关于 GRIB 压缩的看不懂感觉也用不上，偷懒不翻了。

今天来深入了解一下存储浮点数据时如何指定所需的精度，抛弃那些对于精度来说多余的比特。这些多余的比特往往很随机所以不可压缩，导致标准压缩算法的效果有限。需要注意这种操作是一种**有损压缩**。

<!--more-->

实现方法之一是选择所需精度，用它将浮点数转换为一个能用 `scale` 和 `offset` 还原回去的整数。换句话说：

给定浮点数组和浮点精度，找出浮点数 `scale` 和 `offset`，以及最小整数 `n`，使数组中每个值 `F` 满足：

```
UF = scale * P + offset

其中:
    F 是原始的浮点数
    P 是 n 位整数 (打包值)
    UF 是还原的浮点数 (解包值)
同时:
    abs(F - UF) <= precision
```

这里用到的是绝对精度，单位跟数组相同，比如说 0.25 开尔文。

下面是具体实现。给定数组和精度，找出数组的最小值和最大值，然后：

```
nvalues = 1 + Math.ceil((dataMax - dataMin) / (2 * precision))
n = Math.ceil(log2(nvalues))
offset = dataMin
scale = (dataMax - dataMin) / (2^n - 1)
```

让我们来理解一下这段。想象你沿着实数轴观察 `dataMin` 到 `dataMax` 之间的浮点数，从 `dataMin` 开始每隔 `2 * precision` 间隔在轴上做一个标记，并将这个间隔称为数据的**分辨率**。如果数据范围是 0 到 10 K，精度是 0.25 K，你需要每隔 0.5 K 做一个标记，最后得到 21 个标记，`nvalues` 的公式就是在说这个操作。再想象一下你在 `dataMin` 到 `dataMax` 之间挑选任意一个数 `F`，总会有一个标记离 `F` 不到半个间隔（0.25 K），此即前面要求的 `abs(F - UF) <= precision`。

现在我们知道在目标精度内表示浮点数据需要 `nvalues` 个标记，而 `nvalues` 个标记对应整数 0 到 `nvalues - 1`，`n` 个比特能表示 0 到 `2^n - 1` 个整数，所以表示 `nvalues` 个标记正好需要 `log2(nvalues)` 个比特。在前面的例子里 `log2(21) = ln(21) / ln(2) = 4.39`，向上取整得 5。因为 `2^5 = 32`，所以我们最多能表示 32 个标记，包含 21。

如果按上面的方式从 0 到 10 每隔 0.5 K 做标记，并认为这些标记就是 UF 的值，那么：

```
offset = dataMin
scale = 2 * precision
P = round((F - offset) / scale)        (A)
```

用上述公式计算 `P`，当 `F = dataMin` 时 `P = 0`，当 `F = dataMax` 时 `P = 20`。其它 `F` 都在 `dataMin` 到 `dataMax` 之间，所以 `P` 的值只能是 0 到 20 的 21 个整数（包含边界）。

原理讲完了，但还有改进的空间。在我们的例子里 `nvalues = 21` 不是 2 的幂，但比特数只能取整数，`n = 5` 对应 `2^5 = 32` 个标记。也就是说 5 位整数最多表示 32 个打包值，但其中 `(32 - 21) / 32 = 34%` 我们没用到。

虽然比特数只能取整数，没法让 `2^n` 正好等于 `nvalues`，但可以通过降低分辨率的数值（即提高精度）来用上所有的比特位。方法是 `dataMin` 依旧映射到 `P = 0`，但 `dataMax` 映射到 `P` 的最大值，即 `2^n - 1`。为此需要设置

```
scale = (dataMax - dataMin) / (2^n - 1)
```

于是当 `F = dataMax` 时有

```
P = round((F - offset) / scale)
P = round((dataMax - dataMin) / scale)
P = round((dataMax - dataMin) / (dataMax - dataMin) / (2^n - 1))
P = 2^n - 1
```

一个变体是保留 `P` 的一个值，比如说最大值 `2^n - 1` 来表示缺测，此时 `dataMax` 应该映射到 `2^n - 2`，因此：

```
scale = (dataMax - dataMin) / (2^n - 2)    /*保留一个值作为缺测*/
```

(注意：为缺测留出一个值时需要给 `nvalues` 加 1，可能导致比特数 `n` 也跟着增大。)

此时数据的分辨率是多少呢？还记得

```
UF = scale * P + offset
```

`P` 增大 1 时 `UF` 增大 `scale`，所以 `scale` 就是分辨率。又因为 `precision = resolution / 2`，所以在我们新的缩放里精度等于

```
precision = scale / 2 = (dataMax - dataMin) / (2 * (2^n - 2))
```

可以证明新的精度的数值总是小于一开始指定的精度：点数（标记数）提高到 2 的幂后分辨率更小，因而精度更优。不过使用这种扩展的精度是否会影响压缩还有待研究。

> 译注：虽然文中 `P` 是 5 位整数，现实里位数从 8 起步，以 NumPy 为例，常用的是 `np.uint8`、`np.uint16`、`np.uint32` 等。