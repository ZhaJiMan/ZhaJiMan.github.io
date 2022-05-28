---
title: "Python 相关资源汇总（持续更新中）"
date: 2021-11-29
showToc: true
tags:
- python
- 资源
---

简单汇总罗列一下我在网上找到的还不错的 Python 相关资源，包括语言本身以及各种常用库的教程，当然触手可及的官方文档就不收纳了。通通都是免费资源（付费的咱也看不到），分享给有需要的读者。不过互联网资源并非恒久不灭，说不定哪天域名就失效了，或是原作者突然隐藏文章，且看且珍惜吧。

<!--more-->

## Python 语言

[菜鸟教程：Python 3 教程](https://www.runoob.com/python3/python3-tutorial.html)：零起步的中文教程。

[廖雪峰的 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400)：同上，覆盖话题更广。

[A Byte of Python](https://python.swaroopch.com/)：快速上手 Python 的英文教程，适合有一定编程基础的读者。同时存在名为《简明 Python 教程》的中文版。

[Composing Programs](http://www.composingprograms.com/)：UC Berkeley 大学 CS 61A 课程的讲义，以 Python 语言讲解计算机程序的结构和阐释，其中对数据、抽象和函数的讲解鞭辟入里，非常推荐。

[Python 最佳实践指南](https://pythonguidecn.readthedocs.io/zh/latest/)：提供了关于 Python 安装、配置和日常使用的很多实用建议。

[Python Cookbook 第三版](https://python3-cookbook.readthedocs.io/zh_CN/latest/)：非常经典的一本编程技巧合集，中文版可在线阅读。

[What the f*ck Python](https://github.com/leisurelicht/wtfpython-cn)：列举了一些有趣且鲜为人知的 Python 特性，当遇到 bug 时可以来看看是不是中招了。

[Python 工匠](https://github.com/piglei/one-python-craftsman)：作者 piglei 关于 Python 编程技巧和实践的文章合集。

## NumPy、SciPy 和 Pandas

[NumPy Illustrated: The Visual Guide to NumPy](https://betterprogramming.pub/numpy-illustrated-the-visual-guide-to-numpy-3b1d4976de1d)：以图解的方式形象展现了 NumPy 数组的结构和常用用法，特别是强调了行向量、列向量和矩阵的关系，非常值得一读。[知乎](https://zhuanlan.zhihu.com/p/342356377) 上有中文翻译版。

[Python Data Science Handbook](https://jakevdp.github.io/PythonDataScienceHandbook/)：虽然叫手册，但内容编排有条理，带你入门数据科学必备的几个包。

[From Python to Numpy](https://www.labri.fr/perso/nrougier/from-python-to-numpy/)：介绍了很多高级用法和在物理学中的应用。

[Scipy Lecture Notes](https://scipy-lectures.org/)：关于 NumPy 和 SciPy 的讲义，对于 NumPy 的讲解比常见的教程更深入一些。

[SciPy Cookbook](https://scipy-cookbook.readthedocs.io)：介绍了很多 SciPy 的应用场景，附带一些 NumPy 和 Matplotlib 的技巧。

[Joyful-Pandas](http://joyfulpandas.datawhale.club/)：中文教程，因为内容过于详尽以至于读起来反而有点 painful……

[xarray を用いたデータ解析](https://qiita.com/fujiisoup/items/0d71995e54055e9708fc)：xarray 的开发者之一写的日文入门教程。

## Matplotlib

[Anatomy of Matplotlib](https://github.com/matplotlib/AnatomyOfMatplotlib)：很经典的入门教程，比官方 User Guide 更详细一点。

[Cheatsheets for Matplotlib users](https://github.com/matplotlib/cheatsheets/)：速查表，有助于快速查询颜色、色表、线形、散点形状、常用函数等。

[Matplotlib 3.0 Cookbook](https://github.com/apachecn/apachecn-ds-zh/tree/master/docs/matplotlib-30-cookbook)：覆盖了方方面面的技巧，有需要可以查阅。

[Scientific Visualization: Python + Matplotlib](https://github.com/rougier/scientific-visualization-book)：站在科学可视化的高视点指导作图，虽然缺乏操作细节，但书中的示例非常炫酷，代码很值得学习。

[Creating publication-quality figures with Matplotlib](https://github.com/jbmouret/matplotlib_for_papers)：教你如何将出图提升到可出版的质量。

[The Art of Effective Visualization of Multi-dimensional Data](https://towardsdatascience.com/the-art-of-effective-visualization-of-multi-dimensional-data-6c7202990c57)：带你巡游多维数据可视化的种种方法，貌似一些公众号里有中文翻译版。

[The Architecture of Matplotlib](http://aosabook.org/en/matplotlib.html)：创始人关于 Matplotlib 架构的解说，不错的补充材料。

[Image Processing in Python with Pillow](https://auth0.com/blog/image-processing-in-python-with-pillow/)：利用 Pillow（即 PIL）处理图片的入门教程。PIL 在裁剪、拼接和转换格式等方面比 Matplotlib 更方便。

## 气象相关

[Python for Atmosphere and Ocean Scientists](https://carpentries-lab.github.io/python-aos-lesson/)：超入门教程，带你过一遍 xarray、Cartopy、Git 工具链。

[An Introduction to Earth and Environmental Data Science](https://earth-env-data-science.github.io/intro.html)：比上一个更详细一些，推荐。

[Unidata Python Training](https://unidata.github.io/python-training/)：unidata 整的教程，包含 Python 基础和很多气象例子，推荐。

[ATM 623: Climate Modeling](http://www.atmos.albany.edu/facstaff/brose/classes/ATM623_Spring2015/Notes/index.html)：动手学习气候模式相关的知识。

[ATSC 301: Atmospheric radiation and remote sensing](https://clouds.eos.ubc.ca/~phil/courses/atsc301/): 大气辐射和遥感相关的课程讲义，附带需要使用 Python 的练习。

[HDF-EOS: COMPREHENSIVE EXAMPLES](https://hdfeos.org/zoo/index.php)：处理 NASA HDF/HDF-EOS 卫星文件的例子集。

[Project Pyhtia](https://projectpythia.org/)：貌似是 NCAR 搞的在线 training 项目，目前很多教程内容都不全，但其提供的其它网站的 [资源列表](https://projectpythia.org/gallery.html) 非常齐全。

[气象绘图教程合集](https://mp.weixin.qq.com/s/zX9IsuJ_QiH31Hq7P-2Mow)：云台书使在公众号上发布的系统性的气象绘图教程。

[摸鱼的气象](https://space.bilibili.com/9517712)：摸鱼咯在 B 站发布的手把手教学视频。

[气 Py](https://space.bilibili.com/676991774)：老李的系列教学视频，同时他还搬运了很多 MetPy Mondays 的视频。另有配套教材 [气Py_Python气象数据处理与可视化](http://bbs.06climate.com/forum.php?mod=viewthread&tid=101507)。

[气象 Python 学习馆](https://mp.weixin.qq.com/s/HbZUgM-jdTOdYuvKP2CTgA)：深雨露的公众号，制作精良，讲解细致，唯一的缺点可能是数据结构都集中在 xarray 上。

[气象数据科学优质教程&项目集锦](https://www.heywhale.com/mw/project/619328ceb7de000017e4b273)：和鲸社区汇总的多作者的气象教程合集。

[Python空间数据处理实战](https://blog.csdn.net/theonegis/article/details/80089375)：GIS 相关的教程，含有栅格数据和 shapefile 文件的处理。
