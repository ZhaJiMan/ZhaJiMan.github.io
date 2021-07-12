---
title: "Matplotlib 系列：导出高 DPI 的图片"
date: 2021-04-08
showToc: true
math: true
tags:
- matplotlib
---

## 前言

昨天一同学问我怎么把已经画好的图片的 DPI 改到 300，以满足期刊对图片清晰度的要求。上网搜索一番后才发现，虽然我经常在 Matplotlib 中用 `dpi` 参数来调节图片清晰度，但实际上我对这个概念半懂不懂。这次借这个契机调研和总结一下相关的知识。本文将会依次介绍

- 分辨率和 DPI 是什么。
- DPI 和清晰度的关系。
- 如何导出期刊要求的高 DPI 图片。

<!--more-->

## 分辨率

这里的图片指的是位图（bitmap），一张图片由无数个彩色的小像素点组成，Matplotlib 支持的位图格式有 png、jpg、jpeg、png、tiff 等。我们常用分辨率（resolution）来描述图片的大小，例如说一张图片的分辨率是 800 x 400，即指这张图片宽 800 个像素，高 400 个像素。Windows 对一张 jpg 图片打开右键菜单，在“属性”里的“详细信息”里就能看到图片的分辨率，如下图所示

![menu](/matplotlib_dpi/menu.png)

在其它领域里分辨率一词通常描述仪器分辨细节的精细程度，而图片的分辨率仅仅是指图片大小，所以对于图片大小来说，一个更准确的术语是 pixel dimensions。不过既然 Windows 的菜单里都这么显示了，那后文将继续沿用分辨率的说法。

## 尺寸

除了用像素数，图片的尺寸还可以用物理单位来描述，用来指定打印时图片在纸上的大小。例如对于一张分辨率为 800 x 400 的图片，我们希望维持原宽高比打印出来，那么可以设定其尺寸为宽 8 英寸，高 4 英寸（1 英寸约为 2.54 厘米）。这个尺寸可以任意设定，毕竟想打印多大完全由你决定。

## DPI 和 PPI

![PPI_and_DPI](/matplotlib_dpi/PPI_and_DPI.png)

如果说分辨率和尺寸是长度量的话，那么 DPI 和 PPI 就是密度量。它们的定义如下

- DPI（dots per inch）：每英寸长度里含有的打印机墨点数。

- PPI（pixels per inch）：每英寸长度里含有的像素数。

DPI 表现的是打印机的精细程度。对于同样大小的纸张，打印机的 DPI 更高，打印时就会用上更多墨点，那么打印效果自然也更好。

电子设备借鉴了打印设备里 DPI 的概念，用 PPI 来衡量像素点的物理尺寸。PPI 对于显示器和图片的意义稍有不同，下面来分别介绍。首先，显示器的 PPI 计算公式为

$$
\rm{PPI}=对角线像素数/对角线物理长度
$$

给定屏幕大小，PPI 更高则屏幕含有的像素数更多，那么显示效果会更好。例如苹果的 iPhone 就强调其 Retina 屏幕的像素密度高达 326 PPI，有着超出人眼识别能力的细腻效果（广告语看看就得了）。

对图片来说，PPI 和 DPI 这两个术语经常混淆使用，例如 Windows 菜单就称呼图片单位英寸的像素数为 DPI，那么后文也会沿用这一说法。图片 DPI 的计算方法是

$$
\rm{水平DPI}=宽度像素数/物理宽度
$$
$$
\rm{垂直DPI}=高度像素数/物理高度
$$

可以看出，DPI 就是将图片从像素尺寸缩放到物理尺寸的比值。另外，DPI 的倒数即每个像素的单位物理长度，因为我们总是希望像素的物理形状是正方形，所以大多数情况下水平 DPI 就等于垂直 DPI，这样打印出来的图片也能维持原有的宽高比。

一些图片格式会记录图片的 DPI 值，Windows 下图片的右键菜单属性栏里便能看到。我们在对图片进行排版或打印时，软件会根据图片的分辨率和 DPI 自动设定图片的纸上尺寸。不过如果你想把图片打印大点，那么根据定义计算，图片 DPI 会变小；想打印小点，图片 DPI 就会变大——没错，DPI 并不是图片的固有属性，真正决定 DPI 的是图片分辨率和你想要的纸上尺寸，右键菜单属性栏里的数值只是个参考。这一点还可以从两个例子说明，一是 png 格式压根不含 DPI 值，你得根据打印需求自己去算；二是可以用 Pillow 库直接修改图片的 DPI 值

```Python
from PIL import Image

# test1.tif的原始DPI为50
img = Image.open('test1.tif')
img.save('test2.tif', dpi=(300, 300), compression=None)
```

用上面的代码可以把一张特别糊的图片改成 300 DPI 的“出版级”图片，然而图片清晰度和体积一点没变，依旧说明图片元信息（metadata）里的 DPI 值只是个摆设。

## DPI 与清晰度

我们可能听过 DPI 越高越清晰的说法，这里需要明确，DPI 是打印机、显示器，还是图片的 DPI？清晰是指什么东西清晰？

原则上打印机的 DPI 越高，打印出的纸质图片越清晰；显示器的 PPI 越高，显示效果越好。对图片则要分情况讨论。如果给定图片分辨率，DPI 越高，打印出来的纸质图片越小，虽然越小越不容易看出瑕疵，但那也不能说成是打印效果更好。如果给定纸上尺寸，DPI 越高，图片的像素数越多，于是问题转化成了：图片像素越多，就会越清晰吗？

答案是不一定，示意图如下（转自知乎专栏 [影响图像画质的因素：图片的分辨率和像素浅谈](https://zhuanlan.zhihu.com/p/43108622)）

![resolution](/matplotlib_dpi/resolution.jpg)

每一排从右往左，采样分辨率从 50 x 50 降至 1 x 1，清晰度显著下降，说明像素越多越清晰；但第一排到第二排将分辨率用 PS 放大到 10 倍，清晰度并没有显著提高，只是像加了柔和滤镜一样。就我个人的理解，只有在从源头生成图片的过程中才有像素越多越清晰的规律，例如拍照时采样了更多像素点、画画时用更多像素描绘细节等；如果只是对图片进行后处理来增多像素的话就不一定能更清晰，例如各种插值方法。

回到前面的问题，给定纸上尺寸时，DPI 越高图片像素数越多，说明图片本身**很可能**会更清晰，那么在不超出打印机 DPI 水平的前提下，打印出来的纸质图片也很可能更清晰。

## 期刊的 300 DPI 要求

由上一节的讨论，我们便能理解期刊为什么对配图的 DPI 有要求了，因为高 DPI 预示着配图在杂志上的显示效果应该会很好（无论是纸质版还是电子版）。下面以 AGU（美国地球物理学会）对位图的要求为例，用 Matplotlib 演示导出高 DPI 图片的方法。

![AGU_1](/matplotlib_dpi/AGU_1.png)

![AGU_2](/matplotlib_dpi/AGU_2.png)

要求 tif 和 jpg 格式的图片在期刊的纸面尺寸上有 300 - 600 的 DPI，tif 图采用 LZW 压缩，jpg 图选择最高的 quality。1/4 版面大小的图片尺寸是 95 x 115 mm。程序如下

```Python
import matplotlib.pyplot as plt

w = 95 / 10 / 2.54
h = 115 / 10 / 2.54
fig = plt.figure(figsize=(w, h))

fig.savefig('output.tif', dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig('output.jpg', dpi=600, pil_kwargs={'quality': 95})
```

`plt.figure` 函数的 `figsize` 参数要求单位为英寸，所以要先把版面尺寸的单位从毫米换算到英寸。`fig.savefig` 方法里可以直接指定 DPI，压缩方法这种与图片格式相关的参数需要传给 PIL 来实现。最后能得到两张分辨率为 2244 x 2716，600 DPI 的图片。需要注意如果 `dpi` 参数的值太高，生成的图片的分辨率和体积太大。

在 Matplotlib 中，给定 `figsize`，`dpi` 越大，绘制同一个元素时会用到更多像素，所以最后导出的图片会更清晰。此即前面提过的从源头上生成清晰的图片。而后处理增加 DPI 的方法也有：导入 PS 中插值放大；粘贴到 PPT 修改 slide 的分辨率和 DPI，再导出整张 slide；用 AI 把位图转换成矢量图等。后处理方法的问题在于，如果处理前图片就很糊，那么处理后只能得到高 DPI 的假高清图。

当然，最最简单的方式是，从一开始就不要画位图，全部以矢量图的格式导出（eps、pdf 等），这样就完全没有本文中的问题了，所以本文白写了（悲）。

## 额外说明

额外说明一点搜到的实用小知识。

Matplotlib 中的线宽和字体字号是以磅（point）为单位的，有

$$1\ \rm{pt}=1/72\ \rm{inch}$$

例如，`linewidth=72` 时，线宽恰好为 1 英寸。注意这是个物理单位，对应于纸上长度。所以增大 `figsize` 时图中元素会显得更小更细，而增大 `dpi` 时图中元素大小不变，但图片像素更多、显示效果更清晰。

## 参考资料

[Dots per inch - Wikipedia](https://en.wikipedia.org/wiki/Dots_per_inch)

[Relationship between dpi and figure size](https://stackoverflow.com/questions/47633546/relationship-between-dpi-and-figure-size)

[How to ensure your images meet the minimum requirement for printing - DPI explained](https://www.radiologytutor.com/index.php/cases/miscellaneous/63-how-to-ensure-your-images-meet-the-minimum-requirement-for-printing-dpi-explained)

[GRAPHIC REQUIREMENTS - AGU](https://www.agu.org/Publish-with-AGU/Publish/Author-Resources/Graphic-Requirements)

[DPI 和 PPI 的区别是什么？](https://www.zhihu.com/question/23770739/answer/25619192)
[照片的分辨率300dpi那么它的水平分辨率和垂直分辨率分别是多少？](https://www.zhihu.com/question/340341384/answer/789781560)
