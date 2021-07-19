---
title: "二值图像的连通域标记"
date: 2021-07-19
showToc: true
tags:
- 图像处理
---

## 简介

连通域标记（connected component labelling）即找出二值图像中互相独立的各个连通域并加以标记，如下图所示（引自 [MarcWang 的 Gist](https://gist.github.com/MarcWang/5f346375246e1cdb88dbe745b840cfaa)）

![diagram](/connected_component_labelling/diagram.jpg)

可以看到图中有三个独立的区域，我们希望找到并用数字标记它们，以便于计算各个区域的轮廓、外接形状、质心等参数。连通域标记最基本的两个算法是 Seed-Filling 算法和 Two-Pass 算法，下面便来分别介绍它们，并用 Python 加以实现。

<!--more-->

## Seed-Filling 算法

直译即种子填充，以图像中有值的点为种子，然后不断向其它连通区域蔓延，直至将一个连通域完全填满。示意动图如下（引自 [icvpr 的博客](https://blog.csdn.net/icvpr/article/details/10259577)）

![seed-filling](/connected_component_labelling/seed-filling.gif)

具体思路为：循环遍历图像中的每一个像素，如果某个像素有值，且之前未标记过，那么用数字对其进行标记，并寻找它邻近的有值且未被标记过的像素，对这些像素进行标记，并以同样的方法继续寻找邻近像素的邻近像素，并加以标记……如此循环往复，直至将这些互相连通的像素都标记完毕。此即连通域 1，接着继续遍历图像像素，看能不能找到下一个连通域。下面的实现采用的是深度优先搜索（DFS）的策略，将找到的邻近像素压入栈中，弹出栈顶的像素，对其进行标记，再把这个像素的邻近像素压入栈中，重复操作直至栈内再无未标记的像素。

```python
import numpy as np

def get_neighbor_indices(labelled, row, col, connectivity):
    '''找出一点邻域内label值为-1的点的下标.'''
    nrow, ncol = labelled.shape
    if connectivity == 4:
        indices = (
            (row - 1, col),  # 上
            (row, col - 1),  # 左
            (row, col + 1),  # 右
            (row + 1, col)   # 下
        )
    elif connectivity == 8:
        indices = (
            (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),  # 上
            (row, col - 1),                         (row, col + 1),  # 中
            (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)   # 下
        )

    for x, y in indices:
        if x >= 0 and x < nrow and y >= 0 and y < ncol:
            if labelled[x, y] == -1:
                yield x, y

def seed_filling(image, connectivity=4):
    '''
    用Seed-Filling算法寻找图片里的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        二维整型或布尔型数组,0代表图像的背景,1代表前景.

    connectivity : int
        指定邻域为4或8个像素.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol)
        二维整型数组,元素的数值表示所属连通域的标号.
        0表示背景,从1开始表示不同的连通域.

    nlabel : int
        图像中连通域的个数.
    '''
    nrow, ncol = image.shape
    # 用-1表示图像上有值,但还未被标记过的像素.
    labelled = np.asarray(-image, dtype=int)
    label = 1

    for row in range(nrow):
        for col in range(ncol):
            if labelled[row, col] != -1:
                continue
            labelled[row, col] = label
            neighbor_indices = list(get_neighbor_indices(
                labelled, row, col, connectivity
            ))
            # 采用DFS算法,弹出并标记栈顶的像素,再压入其邻域的未被标记过的像素.
            while neighbor_indices:
                neighbor_index = neighbor_indices.pop()
                labelled[neighbor_index] = label
                neighbor_indices.extend(list(get_neighbor_indices(
                    labelled, *neighbor_index, connectivity
                )))
            label += 1

    return labelled, label - 1
```

## Two-Pass 算法

顾名思义，是会对图像过两遍循环的算法。第一遍循环先粗略地给有值的像素进行标记，第二遍循环中再根据不同标签之间的关系对第一遍的结果进行修正。示意动图如下（引自 [icvpr 的博客](https://blog.csdn.net/icvpr/article/details/10259577)）

![two-pass](/connected_component_labelling/two-pass.gif)

具体思路为

- 第一遍循环时，如果一个像素有值，且它周围的像素都是 0，那么它就是一个新的独立区域，需要赋予其一个新标签。如果这个有值的像素周围有其它像素，则说明它们之间互相连通，此时随便用它们中的一个旧标签值来标记当前像素即可，同时要用并查集记录这些像素标签间的关系。
- 因为我们总是只利用了当前像素邻域的信息（考虑到循环方向是从左上到右下，其实当前像素右下方的信息也是利用不到的），所以第一遍循环进行的标记是有问题的，虽然有值的区域都会被标记，但同一个连通域内的像素很可能有不同的标签值。不过利用第一遍循环时获得的标签之间的关系（记录在并查集中），可以在第二遍循环中将同属一个集合（连通域）的不同标签修正为同一个标签。
- 根据第二遍循环中合并标签操作的策略的不同，最后得到的一系列的标签值可能是不连续的，依据需求可以进行第三遍循环，将标签值替换为连续的排名值（rank）。

其中提到的并查集是一种处理不相交集合的数据结构，支持查询元素所属、合并两个集合的操作。利用它就能处理标签和连通域之间的从属关系。我是看 [算法学习笔记(1) : 并查集](https://zhuanlan.zhihu.com/p/93647900) 这篇知乎专栏学的。下面的实现中利用负值存储根节点的秩（即树的深度），同时考虑到并查集的大小会随循环逐步增大，所以没有写成通用的类。

```python
import numpy as np

def get_neighbor_labels(labelled, row, col, connectivity):
    '''找出一点上边和左边大于零的label.'''
    nrow, ncol = labelled.shape
    if connectivity == 4:
        indices = (
            (row - 1, col),     # 上
            (row, col - 1)      # 左
        )
    elif connectivity == 8:
        indices = (
            (row - 1, col - 1), # 左上
            (row - 1, col),     # 上
            (row - 1, col + 1), # 右上
            (row, col - 1)      # 左
        )
    else:
        raise ValueError('connectivity must be 4 or 8')

    for x, y in indices:
        if x >= 0 and x < nrow and y >= 0 and y < ncol:
            neighbor_label = labelled[x, y]
            if neighbor_label > 0:
                yield neighbor_label

def find(uf, i):
    '''递归查找并查集uf中第i个节点的根节点,同时压缩路径.'''
    parent = uf[i]
    if parent < 0:
        return i
    else:
        root = find(uf, parent)
        uf[i] = root
        return root

def union(uf, i, j):
    '''按秩合并i和j所属的两个集合.'''
    root_i, root_j = find(uf, i), find(uf, j)
    rank_i, rank_j = -uf[root_i], -uf[root_j]
    if root_i == root_j:
        return None
    else:
        if rank_i > rank_j:
            uf[root_j] = root_i
        elif rank_i < rank_j:
            uf[root_i] = root_j
        else:
            uf[root_i] = root_j
            uf[root_j] -= 1

def roots(uf):
    '''获取并查集中所有节点的根节点.'''
    return [find(uf, i) for i in range(len(uf))]

def two_pass(image, connectivity=4):
    '''
    用Two-Pass算法寻找图片里的连通域.

    Parameters
    ----------
    image : ndarray, shape (nrow, ncol)
        二维整型或布尔型数组,0代表图像的背景,1代表前景.

    connectivity : int
        指定邻域为4或8个像素.

    Returns
    -------
    labelled : ndarray, shape (nrow, ncol)
        二维整型数组,元素的数值表示所属连通域的标号.
        0表示背景,从1开始表示不同的连通域.

    nlabel : int
        图像中连通域的个数.
    '''
    nrow, ncol = image.shape
    labelled = np.zeros_like(image, dtype=int)
    uf = [-1]    # 初始化并查集.
    label = 1

    # 第一遍循环,用label标记出连通的区域.
    for row in range(nrow):
        for col in range(ncol):
            if image[row, col] == 0:
                continue
            # 若左边和上边没有label大于零的像素,则当前像素获得新label.
            # 否则用并查集记录相邻像素的label的关系.
            neighbor_labels = list(get_neighbor_labels(
                labelled, row, col, connectivity
            ))
            if len(neighbor_labels) == 0:
                labelled[row, col] = label
                uf.append(-1)
                label += 1
            else:
                first_label = neighbor_labels[0]
                labelled[row, col] = first_label
                for neighbor_label in neighbor_labels[1:]:
                    union(uf, first_label, neighbor_label)

    # 获取代表每个集合的label,并利用大小排名重新赋值.
    labels = np.array(roots(uf), dtype=int)
    values = np.unique(labels)
    for i, value in enumerate(values):
        labels[labels == value] = i
    # 第二遍循环赋值利用ndarray的advanced indexing实现.
    labelled = labels[labelled]

    return labelled, i
```

如果不介意对 SciPy 的依赖，重新排名的部分可以用 `scipy.stats.rankdata(method='dense')` 来实现。Two-Pass 算法更复杂一些，但因为不需要进行递归式的填充，所以理论上要比 Seed-Filling 更快。

## 其它方法

实际应用中推荐使用 `scipy.ndimage.label` 或 `skimage.measure.label` 函数，因为它们底层都是用 Cython 实现的，所以速度秒杀前面的手工实现。如果懂 OpenCV 的话，还可以调用 `cv2.connectedComponents` 函数。我完全不懂 OpenCV，就不介绍了。

## 例子

以一个随机生成的 100*100 的二值数组为例，测试 `scipy.ndimage.label`、Two-Pass 实现和 Seed-Filling 实现的效果，采用 8 邻域连通，效果如下图

![random](/connected_component_labelling/random.png)

可以看到三种方法都找出了 16 个连通域，其中 `scipy.ndimage.label` 与 Seed-Filling 连标签顺序都是一模一样（填色相同）的，而 Two-Pass 的标签顺序可能受并查集的合并策略影响而有所差异——不过标签顺序意义不大就是了。下面再以一个更复杂 800*800 大小的空露露图片为例

![image](/connected_component_labelling/image.png)

将图片二值化后再进行连通域标记，可以看到おつるる的字样被区分成多个区域，猫猫和露露也都被识别了出来。代码如下

```python
import numpy as np
from PIL import Image
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt

from connected_components import two_pass, seed_filling

if __name__ == '__main__':
    # 将测试图片二值化.
    picname = 'ruru.png'
    image = Image.open(picname)
    image = np.array(image.convert('L'))
    image = ndimage.gaussian_filter(image, sigma=2)
    image = np.where(image < 220, 1, 0)

    # 设置二值图像与分类图像所需的cmap.
    cmap1 = mpl.colors.ListedColormap(['white', 'black'])
    white = np.array([1, 1, 1])
    cmap2 = mpl.colors.ListedColormap(
        np.vstack([white, mpl.cm.tab20.colors])
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 关闭ticks的显示.
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # 显示二值化的图像.
    axes[0, 0].imshow(image, cmap=cmap1, interpolation='nearest')
    axes[0, 0].set_title('Image', fontsize='large')

    # 显示scipy.ndimage.label的结果.
    # 注意imshow中需要指定interpolation为'nearest'或'none',否则结果有紫边.
    s = np.ones((3, 3), dtype=int)
    labelled, nlabel = ndimage.label(image, structure=s)
    axes[0, 1].imshow(labelled, cmap=cmap2, interpolation='nearest')
    axes[0, 1].set_title(
        f'scipy.ndimage.label ({nlabel} labels)', fontsize='large'
    )

    # 显示Two-Pass算法的结果.
    labelled, nlabel = two_pass(image, connectivity=8)
    axes[1, 0].imshow(labelled, cmap=cmap2, interpolation='nearest')
    axes[1, 0].set_title(f'Two-Pass ({nlabel} labels)', fontsize='large')

    # 显示Seed-Filling算法的结果.
    labelled, nlabel = seed_filling(image, connectivity=8)
    axes[1, 1].imshow(labelled, cmap=cmap2, interpolation='nearest')
    axes[1, 1].set_title(f'Seed-Filling ({nlabel} labels)', fontsize='large')

    fig.savefig('image.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
```

最后说下速度，`scipy.ndimage.label` 比 Two-Pass 快几百倍，而 Two-Pass 只比 Seed-Filling 快一倍。处理分辨率大一点的图片时，后两者的速度有点急人。可能是因为纯 Python 实现确实太慢（毕竟完全没用上 NumPy 的向量性），或者我前面写的代码太烂。还请懂行的读者指点一下。

## 参考链接

网上很多教程抄了这篇，但里面 Two-Pass 算法的代码里不知道为什么没用并查集，可能会有问题。

[OpenCV_连通区域分析（Connected Component Analysis-Labeling）](https://blog.csdn.net/icvpr/article/details/10259577)

一篇英文的对 Two-Pass 算法的介绍，Github 上还带有 Python 实现。

[Connected Component Labelling](https://jacklj.github.io/ccl/)

代码参考了

[你都用 Python 来做什么？laiyonghao 的回答](https://www.zhihu.com/question/20799742/answer/1739070110)

[连通域的原理与Python实现](https://zhuanlan.zhihu.com/p/97689424)