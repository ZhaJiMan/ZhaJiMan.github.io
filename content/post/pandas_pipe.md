---
title: "Pandas 系列：管道风格"
date: 2022-10-29
showToc: true
tags:
- python
- pandas
---

## R 语言的管道

这回来介绍一下如何利用管道（pipe）风格将 Pandas 相关的代码写得更易读，不过首先让我们看看隔壁 R 语言中管道是怎么用的。假设输入是 `x`，经过连续四个函数的处理后得到输出 `y`，代码可以按顺序写：

```R
x1 <- func1(x, arg1)
x2 <- func2(x1, arg2)
x3 <- func3(x2, arg3)
y <- func4(x3, arg4)
```

<!--more-->

流程很清晰，但函数与函数之间会产生中间变量。这里为了方便取 `x` 加数字后缀形式的名字，日常编程时最好还是起个有意义点的名字，例如 `x_after_func1` 之类的。另一种简练的写法是：

```R
y <- func4(func3(func2(func1(x, arg1), arg2), arg3), arg4)
```

代码更短，也没有中间变量了，但代价是重看代码时需要像剥洋葱一样从两边向中间一层层读。并且当函数名更长参数更多时，可读性会进一步恶化，列数也很容易超出屏幕的宽度。

这样看来似乎第一种风格更为妥当。不过，若是活用 magrittr 包里的管道符 `%>%` 的话，就能写出既清晰又简练的代码了。简单介绍一下 `%>%` 的功能：

- `x %>% f` 等价于 `f(x)`。
- `x %>% f(y)` 等价于 `f(x, y)`。
- `x %>% f(y, .)` 等价于 `f(y, x)`。
- `x %>% f(y, z = .)` 等价于 `f(y, z = x)`。

即输入 `x` 通过管道 `%>%` 传给函数 `f`，`f` 里不用写 `x`，管道会自动把 `x` 作为 `f` 的第一个参数；如果 `x` 并非第一个参数，那么可以用占位符 `.` 代指 `x`。

应用了管道符后的代码风格是：

```R
y <- x %>%
  func1(arg1) %>%
  func2(arg2) %>%
  func3(arg3) %>%
  func4(arg4)
``` 

格式整齐，代码顺序和操作顺序一致，语义清晰，没有多余的中间变量，强迫症患者感到十分舒适。这种写法的另一个好处是，增删函数就像增删空行一样简单，而前两种风格改起来就会十分烦人。

## Pandas 中的管道

遗憾的是 Python 中并没有成熟的管道包，但有一种神似的写法：

```Python
x = 'fried chicken\n'
y = x.rstrip().replace('fried', 'roast').upper().rjust(20)
print(y)
```

```
       ROAST CHICKEN
```

即对 `x.rstrip()` 方法返回的字符串调用 `replace` 方法，再对返回值调用 `upper` 方法，最后调用 `rjust` 方法，构成了方法链（method chaining）。这个写法看似简洁，实则局限很大：以一节节管道做比喻的话，R 中每节管子可以是任意函数，而 Python 中每节管子只能是输入管子的对象自带的方法。如果你想实现的操作不能用输入对象的方法达成，那么管道就连不起来，你还是得乖乖打断管道，在下一行调用函数或写表达式。

但细分到用 Pandas 包做数据分析的领域，基于方法链的管道已经完全够用了：绝大部分操作都可以用 `DataFrame` 或 `Series` 的方法实现，并且方法返回的结果依旧是 `DataFrame` 或 `Series` 对象，保证可以接着调用方法；外部函数用 `map`、`apply`、`applymap` 或 `pipe` 方法应用到数据上。下面以处理站点气象数据表格为例：

- 查询指定站点。
- 丢弃站点列。
- 将时间列转为 `DatetimeIndex`。
- 按时间排序。
- 去除时间上重复的记录。
- 设置时间索引。
- 将 999999 替换成 NaN。
- 重采样到逐小时分辨率并插值填充。
- 加入风速分量列。

先来个普通风格：

```Python
def wswd_to_uv(ws, wd):
    '''风速风向转为uv分量.'''
    wd = np.deg2rad(270 - wd)
    u = ws * np.cos(wd)
    v = ws * np.sin(wd)
    
    return u, v

station = 114514
df.query('station == @station', inplace=True)
df.drop(columns='station', inplace=True)
df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M')
df.sort_values('time', inplace=True)
df.drop_duplicates(subset='time', keep='last', inplace=True)
df.set_index('time', inplace=True)
df.mask(df >= 999999, inplace=True)
df = df.resample('H').interpolate()
df['u'], df['v'] = wswd_to_uv(df['ws'], df['wd'])
```

得益于很多方法自带原地修改的 `inplace` 参数，中间变量已经很少了。再来看看管道风格：

```Python
def set_time(df, fmt):
    return df.assign(time=pd.to_datetime(df['time'], format=fmt))

def add_uv(df):
    u, v = wswd_to_uv(df['ws'], df['wd'])
    return df.assign(u=u, v=v)

dfa = (df
    .query('station == @station')
    .drop(columns='station')
    .pipe(set_time, fmt='%Y-%m-%d %H:%M')
    .sort_values('time')
    .drop_duplicates(subset='time', keep='last')
    .set_index('time')
    .mask(lambda x: x >= 999999)
    .resample('H').interpolate()
    .pipe(add_uv)
)
```

个人感觉管道风格的格式更整齐，一眼就能看出每行的“动词”（方法）。去除了每行都有的 `inplace` 参数后，不仅视觉上更清爽，还保证了一套操作下来输入数据不会无缘无故遭到修改。接着再说说管道风格里的两个细节。

### pipe

就是 Pandas 版的 `%>%`：

- `df.pipe(func)` 等价于 `func(df)`。
- `df.pipe(func, *args, **kwargs)` 等价于 `func(df, *args, **kwargs)`。
- `df.pipe((func, 'arg2'), arg1=a)` 等价于 `func(arg1=a, arg2=df)`。

可以将复杂的多行运算打包成形如 `func(df, *args, **kwargs)` 的函数，然后结合 `pipe` 使用。前文的 `set_time` 和 `add_uv` 函数就是例子。

### assign

`assign` 方法的功能就是无副作用的列赋值：复制一份对象自己，在列尾添加新列或是修改已有的列，然后返回这份拷贝：

```Python
# 相当于:
# dfa = df.copy()
# dfa['a'] = a
# dfa['b'] = b
dfa = df.assign(a=a, b=b)

# 相当于:
# df['a'] = a
# df['b'] = b
df.assign(a=a, b=b, inplace=True)
```

第一次看到 `assign` 时我只觉得多此一举，赋值不是用等号就可以吗？但后来我意识到它是搭配管道风格使用的：想要对管道内的中间变量做列赋值，同时不中断管道，就只能用 `assign` 方法。同时考虑到中间变量里的内容可能已经跟原始输入大不相同，`assign` 的参数还可以是以调用对象本身（即 `self`）为唯一参数的函数：

```Python
# 省略号表示略去的方法.
dfa = (df
    ...
    .assign(u=uwind, v=vwind)
    .assign(ws=lambda x: np.hypot(x['u'], x['v']))
    ...
)
```

这里不能写成 `assign(ws=np.hypot(df['u'], df['v']))`，因为 `df` 里本来是没有 `u` 和 `v` 的，但中间变量有，那么把匿名函数传给 `assign` 就可以解决这一问题。

不只是 `assign`，`where` 和 `mask` 等方法，乃至 `loc` 和 `iloc` 索引器都能接受函数（准确来说是 `callable` 对象），方便在管道风格中使用。

## 什么时候该用管道

管道并非优雅代码的万金油，而是有特定使用场景的：

- 输入经过一连串的操作得到一个输出的情况适合使用管道，输入和输出都很多时显然不太适合。

- 管道里的操作多于十个时会使 debug 变得很麻烦，因为缺少中间变量来定位 bug。建议当操作很多时适当分出中间变量，不要一个管道写到头。

- 方法链中对象的类型发生改变时建议将链条进行拆分，不然会令人迷惑。


## 参考链接

[A Forward-Pipe Operator for R • magrittr](https://magrittr.tidyverse.org/)

[R for Data Science: 18 Pipes](https://r4ds.had.co.nz/pipes.html)

[pandas 在使用时语法感觉很乱，有什么学习的技巧吗？](https://www.zhihu.com/question/289788451/answer/2495499460)