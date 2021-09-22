---
title: "Numpy 系列：random 模块的变化"
date: 2021-09-21
showToc: true
tags:
- numpy
---

## 前言

这几天要用 NumPy 生成随机数，所以去查了一下 `np.random` 模块的官方文档，却惊讶地发现里面介绍的用法跟我的记忆有很大出入：例如以前用 `np.random.rand` 便能生成 [0, 1) 之间均匀分布的随机数，现在文档里记载的却是面向对象风格的写法（创建随机数生成器再调用方法……）。调查一番后发现原来这一改动发生于 NumPy 1.17 版本（2020 年 1 月），并且网上对此的中文介绍也比较少，所以现撰文简单介绍一下该模块在改动前后的两套用法。

<!--more-->

## 原理

先概括一下计算机生成随机数的原理，方便后面理解程序的行为。我们先给定一个用整数表示的随机种子（seed），然后计算机会根据特定的算法（平方取中、线性同余等……）对这个种子不断进行计算，得到一串数字序列。由于输入是确定的，算法的步骤也是完全固定的，所以结果也是唯一确定的——即一个种子对应一个序列。这个序列虽然是完全确定的，但它本身与真实世界中随机过程产生的序列很相似，序列中的每个数字像是随机出现的，且分布接近于均匀分布。于是我们便把这个算法生成的“伪随机序列”当作随机序列来用，再根据需求通过数学变换把均匀分布的随机序列变换为其它概率分布的随机序列。

不过这一做法的缺陷是，若种子不变，那么每次生成的随机序列总是一模一样的，甚至还可以从序列的排列规律中反推出种子的值。为了避免这种情况，可以用操作系统的时间戳或熵池（系统收集的各个设备的环境噪音）信息作为种子，以保证每次运行都产生不同的结果。

更详细的解说请参考 [混乱中的秩序——计算机中的伪随机数序列](https://zhuanlan.zhihu.com/p/33903430) 这篇知乎专栏。我们将会看到，无论是旧版还是新版，`numpy.random` 模块都是按照这一节的流程来生成随机数的。

## 旧版本

### RandomState

虽然我们常用的是 `np.random.rand` 这样的函数命令，但要把用法讲清楚，还是需要从 `RandomState` 类开始。`RandomState` 是 `np.random` 模块中表示随机数生成器的类，内部采用 Mersenne Twister 算法的 MT19937 实现来生成伪随机序列（算法原理在前面提到的专栏中有介绍）。在创建对象时需要指定随机种子，然后通过调用方法来生成其它概率分布的随机数，例如

```python
import numpy as np
from numpy.random import RandomState

seed = 0
rs = RandomState(seed)

# 生成3个[0,1)范围内均匀分布的随机数
print(rs.rand(3))
# 生成3个服从标准正态分布的随机数
print(rs.randn(3))
```

种子可以是一个大于等于 0 的整数，也可以是这样的整数构成的一维序列。无论种子是哪种形式，只要每次给定相同的种子，那么随机数生成器都会生成相同的随机序列，调用方法时会不断从这个序列中抽取数字来进行变换，进而生成相同的随机数。例如

```python
# 生成三个[0,10]范围内的随机整数
rs1 = RandomState(1)
print('seed=1:', rs1.randint(0, 11, 6))

rs2 = RandomState(1)
print('seed=1:', rs2.randint(0, 11, 3), rs2.randint(0, 11, 3))

rs3 = RandomState(2)
print('seed=2:', rs3.randint(0, 11, 6))
```

结果为

```
seed=1: [5 8 9 5 0 0]
seed=1: [5 8 9] [5 0 0]
seed=2: [8 8 6 2 8 7]
```

可以看到当种子都为 1 时，两个不同的 `RandomState` 对象生成的随机数相同（尽管 `rs2` 调用了两次方法）；但当种子为 2 时，结果便发生了变化。下面再举一个用时间戳作为种子的例子

```python
import time

seed = int(time.time())
rs = RandomState(seed)
for _ in range(3):
    print(rs.randint(0, 11, 3))
```

注意不要把设置种子的语句写在循环里，因为取整后的时间戳的间隔只有 1 秒，而循环一次的速度一般远快于 1 秒，这就导致循环内一直使用同一个种子，最后产生三组一模一样的随机数。其实，在创建 `RandomState` 对象时如果不给出种子（即默认的 `seed=None`），那么程序会自动利用熵池和时间信息来确定种子的值。所以总结一下就是，如果你需要程序结果是可复现的（reproducible），那么使用固定种子即可；如果你需要每次都使用不同的随机数，那么大胆写上 `rs = RandomState()` 即可。

下面用表格总结一下 `RandomState` 对象常用的方法

|   方法    |                             效果                             |
| :-------: | :----------------------------------------------------------: |
|  `rand`   | 生成 [0, 1) 范围内均匀分布的浮点随机数。本质是 `random_sample` 包装后的版本。 |
| `randint` |      生成 [low, high) 范围内离散均匀分布的整型随机数。       |
|  `randn`  | 生成服从标准正态分布的随机样本。对于更一般的正态分布可以使用 `normal` 方法。 |
| `choice`  |                对给定的一维数组进行随机抽样。                |

### 调用函数

对我们更为熟悉的可能是直接调用函数的用法，例如

```python
np.random.seed(1)
print(np.random.rand(3))
print(np.random.randint(0, 11, 3))
print(np.random.randn(3))
```

大家很容易看出其用法与上一节大差不差，所以就不详细解说了。联系在于，首次调用函数时，NumPy 会偷偷在全局创建一个 `RandomState` 对象，然后用这个对象来生成随机数，作为这些函数的返回值。所以调用函数只是一种偷懒（handy）的用法罢了。这种用法的缺点很明显，如果代码中有地方改动了种子，会影响全局的随机数结果，更别说在并行时还可能出现同时修改种子的情况。尽管有着明显的缺点，但在 `np.random` 模块大改之前，官方文档和各路教程都主推这一用法，我们在使用时需要多加小心。

## 新版本

1.17 版本前 `np.random` 中存在面向对象和调用函数两种用法，而 1.17 版本后则统一使用新的面向对象式的用法，并在功能和性能方面作出了很多改进，下面便来一一解说。首先新版本为了能支持使用不同的随机数生成算法，将原先的 `RandomState` 细分为两个类：`BitGenerator` 和 `Generator`。前者通过随机数生成算法产生随机序列，后者则对随机序列进行变换。例如

```python
# MT19937和PCG64都是内置的BitGenerator
from numpy.random import MT19937, PCG64, Generator

# BitGenerator接收seed为参数
seed = 1
rng1 = Generator(MT19937(seed))
rng2 = Generator(PCG64(seed))

# 生成3个[0, 10]范围的整数
print(rng1.integers(0, 10, 3, endpoint=True))
print(rng2.integers(0, 10, 3, endpoint=True))
```

结果为

```
[2 9 8]
[5 5 8]
```

新用法的模式与 `RandomState` 非常类似，但 `RandomState` 只支持 Mersenne Twister 算法，而新用法通过更换 `BitGenerator` 对象可以换用不同的随机数生成算法。可以看到尽管种子相同，但不同算法的结果是不一样的。一般来说我们不需要自己选取算法，使用默认的随机数生成器即可。例如

```python
from numpy.random import default_rng

# 等价于 rng = Generator(PCG64())
# 不给定种子时,自动根据熵池或时间戳选取种子
rng = default_rng()
print(rng.integers(0, 11, 3, endpoint=True))
```

默认生成器使用 2014 年提出的 PCG 算法，其性能与统计特性要比 1997 年提出的 Mersenne Twister 算法提高不少。下面用表格总结一下 `Generator` 对象常用的方法

|       方法        |                             效果                             |
| :---------------: | :----------------------------------------------------------: |
|     `random`      | 生成 [0, 1) 范围内均匀分布的浮点随机数。类似于标准库的 `random.random` 。 |
|    `integers`     | 生成 [low, high) 范围内内离散均匀分布的整型随机数。相比 `randint`，增加了指定区间是否闭合的 `endpoint` 参数。 |
| `standard_normal` | 生成服从标准正态分布的随机样本。对于更一般的正态分布可以使用 `normal` 方法。 |
|     `choice`      |                对给定的多维数组进行随机抽样。                |

可以看到 `Generator` 的方法名相比 `RandomState` 更符合直觉，功能上也作出了改进。虽然现在官方推荐新版本的用法，但出于兼容性的考虑，旧版本的用法也依然可以使用。值得注意的是，即便使用相同的随机数生成算法和相同的种子，新版本与旧版本产生的随机数也不会相同，例如

```python
import numpy as np
from numpy.random import RandomState, MT19937, Generator

seed = 1
rs = RandomState(seed)
rng = Generator(MT19937(seed))

decimals = 2
print('RandomState:', np.around(rs.rand(3), decimals))
print('Generator:', np.around(rng.random(3), decimals))
```

结果为

```
RandomState: [0.42 0.72 0.  ]
Generator: [0.24 0.73 0.56]
```

这是因为 `Generator` 在接受种子后还会在内部自动通过 `SeedSequence` 类对种子进行进一步的处理，利用新的散列算法将用户给出的低质量种子转化成高质量种子，以提高生成的随机数的质量。例如对于 Mersenne Twister 算法，如果给出相邻的两个整数种子，那么生成的两串随机序列将会有很大的相似性——即两串序列不够独立。而新引入的 `SeedSequence` 类就能让相邻的种子对应于迥然的两个生成器状态。同时 `SeedSequence` 类还有助于在并行生成随机数时为每个子进程设置相互独立的状态，有需求的读者请参考官方文档 [Parallel Random Number Generation](https://numpy.org/doc/stable/reference/random/parallel.html)，这里就不多加介绍了。当然，即便种子经过了更复杂的处理，原理中提到的种子能决定随机数结果的规则依旧是不变的。

基本用法的介绍就这些，新旧版本的其它差别在官网也有总结（[What’s New or Different](https://numpy.org/doc/stable/reference/random/new-or-different.html)），希望本文能对读者有所帮助。

## 参考链接

[NumPy: Random sampling](https://numpy.org/doc/stable/reference/random/index.html)

[NumPy: Legacy Random Generation](https://numpy.org/doc/stable/reference/random/legacy.html)

[numpy-random函数](https://segmentfault.com/a/1190000016097466)

[NumPy Random Seed, Explained](https://www.sharpsightlabs.com/blog/numpy-random-seed/)

[numpy.randomのGeneratorをためしてみる](https://qiita.com/hnakano863/items/2a959e5731ef5c9191a6)

[Good practices with numpy random number generators](https://albertcthomas.github.io/good-practices-random-number-generators/)

[随机数大家都会用，但是你知道生成随机数的算法吗？](https://zhuanlan.zhihu.com/p/273230064)

[What numbers that I can put in numpy.random.seed()?](https://stackoverflow.com/questions/36847022/what-numbers-that-i-can-put-in-numpy-random-seed)
