---
title: "Python 中操作文件和目录的路径"
date: 2021-03-26
showToc: true
tags:
- python
---

## 前言

之前在 Linux 上用 Python 处理系统的文件和目录时，我都是简单粗暴地用 `os.system` 函数直接执行 shell 命令来实现的。例如新建一个目录并把文件移动进去，我会这么写

```Python
dirpath = './result'
filepath = './data.txt'
os.system(f'mkdir {dirpath}')
os.system(f'mv {filepath} {dirpath}')
```

即把 shell 命令硬编码到程序中。但最近在 Windows 上运行老程序时，因为 `os.system` 默认调用 CMD，所以这种写法的老代码全部木大。

其实借助 Python 标准库中用于系统交互和路径处理的模块，就能尽可能降低代码对平台的依赖，并且模块中也提供有许多方便的函数。本文会记录那些最常用的功能。

<!--more-->

## 基础知识

首先明确一些基础知识，以免后面发生混淆。目录（directory）即我们常说的文件夹，能够存放文件和其它目录。而路径（path）是用于标识文件和目录在文件系统中具体位置的字符串，路径的末尾是文件或者目录的名字，而前面则是一级一级的父目录，每一项通过路径分隔符隔开。

Linux 和 Mac 的路径分隔符是正斜杠 `/`，而 Windows 用的是反斜杠 `\`。在 Python 的字符串中，因为反斜杠还有转义的作用，所以要么用 `\\` 表示一个反斜杠 ，要么使用 raw 字符串（不过以反斜杠结尾时会引起语法解析的错误）。例如

```Python
# Linux下的路径
dirpath = './a/b/c'
# Windows下的路径
dirpath1 = './/a//b//c'
dirpath2 = r'./a/b/c'
```

注意虽然程序中字面值是 `\\`，但打印或输出时是正常的 `\`。其实现在的 Windows 内核兼容正斜杠的写法，在 Python 程序中我们完全可以只使用正斜杠（甚至混用都没问题）。

下面再来谈一谈目录的路径结尾是否该加上斜杠的问题。有些人习惯在目录的路径结尾再添上一个斜杠，以显示这个路径表示的是一个目录而不是文件，并且之后在进行字符串连接时也不必手动插入斜杠。在绝大多数情况下，加或不加并不会影响到命令行的行为。

考虑到 Python 中许多函数在处理路径时会自动去掉结尾的斜杠，以免影响路径的分割（`os.path.basename`、`os.path.dirname` 等函数），本文中不会在结尾加上斜杠。

## os

这个模块提供一些与操作系统进行交互的函数，例如创建和删除目录等。

`os.sep`：属性，值是系统所用的路径分隔符的字符串。

`os.getcwd`：获取工作目录的路径。

`os.chdir`：切换工作目录，功能同 shell 中的 `cd` 命令。

`os.listdir`：返回指定的目录（默认是工作目录）下所有文件和目录的名字组成的列表。注意列表元素的顺序是任意的（尽管我们的运行结果可能是有序的）。

`os.walk`：自上而下遍历一棵目录树，每到一个目录时 yield 一个 `(dirpath, dirnames, filenames)` 的三元组。其中 `dirpath` 是该目录的路径，`dirnames` 是该目录下子目录名字组成的列表，`filenames` 是该目录下文件名组成的列表。下面举个找出目录下所有文件的例子

```Python
def get_all_filepath(dirpath):
    for dirpath, dirnames, filenames in os.walk(dirpath):
        for filename in filenames:
            yield os.path.join(dirpath, filename)
```

`os.mkdir`：创建一个目录。

`os.makedirs`：递归地创建一个目录，即就算我们给出的路径中含有尚不存在的目录，系统也能顺便给创建了。

`os.rmdir`：删除一个空目录，如果目录非空则会报错。

`os.removedirs`：递归地删除空目录。即根据路径从右往左逐个删，碰到非空的目录时就会停下（不然那不得把你根目录给端了）。

`os.remove`：删除一个文件。如果路径指向目录的话会报错。

`os.rename`：给文件或目录重命名。如果重命名到另一个目录下面，就相当于剪切。当目标路径已经存在时，会有比较复杂的行为，建议不要这么做。

`os.replace`：相当于 `os.rename`，但当目标路径指向已经存在的目录时会报错，指向文件时则会直接替换。

`os` 模块中关于文件和目录的常用函数差不多就这些。你可能会问，怎么删除目录的函数都只能作用于空目录，那非空的目录怎么办？这就需要用到更高级的文件操作库——`shutil`。

## shutil

这个模块提供正经的文件/目录的复制、剪切、删除操作。

`shutil.copyfile`：复制文件，要求两个参数都为文件路径。

`shutil.copy`：同样是复制文件，但目标路径可以为目录，这样相当于保持文件名不变复制过去。

`shutil.copytree`：顾名思义，直接复制一整棵目录树，即复制非空的目录。

`shutil.rmtree`：删除一整棵目录树。

`shutil.move`：将文件或非空目录移动到目标目录下面。

## glob

这个模块的功能非常单纯：提供 Unix shell 风格的路径搜索。即可以用通配符实现灵活的匹配，又能直接拿到文件和目录的路径，方便操作。

`glob.glob`：给出含通配符的路径，将与之匹配的路径汇集成列表返回。因为这个函数内部是由 `os.listdir` 实现的，所以也不能保证结果的顺序。Python 3.5 以后提供 `recursive` 选项，指定是否进行递归搜索，用 `**` 匹配目录下的所有内容。

一些例子如下

```Python
# 得到路径dirpath下的文件和目录的路径
glob.glob(os.path.join(dirpath, '*'))
# 得到路径dirpath下所有py文件的路径
glob.glob(os.path.join(dirpath, '**', '*.py'), recursive=True)
```

如果给出的路径是相对路径，那么结果也会是相对路径，绝对路径同理。

如果希望搜索的结果有序排列，可以用列表的 `sort` 方法或 `sorted` 函数进行排序。下面举个搜索路径下所有图片，并按文件名排序的例子

```Python
dirpath = './pics'
filepaths = glob.glob(os.path.join(dirpath, '*.png'))
filepaths.sort(key=lambda x: os.path.basename(x))
```

如果需要节省内存，`glob` 模块还提供返回生成器的 `glob.iglob` 函数。

## os.path

这个模块提供许多处理路径的函数，其实在前面的例子中已经出现过好几次了。

`os.path.normpath`：将路径规范化。能将多余的分隔符去掉，例如 `A//B` 、`A/B/` 和 `A/./B` 都会变成 `A/B`。可以看出，结尾有斜杠对于 Python 来说是不“规范”的。Windows 系统下还会将路径中的正斜杠都替换成反斜杠。

`os.path.abspath`：将路径转换为规范的绝对路径。

`os.path.relpath`：将路径转换为规范的相对路径。

`os.path.basename`：返回路径的基名（即文件或目录的名字）。需要注意，如果路径结尾有斜杠，那么会返回空字符串。

`os.path.dirname`：返回路径的父目录。需要注意，如果路径结尾有斜杠，那么返回的就只是去掉末尾斜杠的路径。

`os.path.splitext`：输入一个文件路径，返回一个二元组，第二个元素是这个文件的扩展名（含 `.`），第一个元素就是扩展名前面的路径。如果路径不指向文件，那么第二个元素会是空字符串。

`os.path.exists`：判断路径是否存在。

`os.path.isfile`：判断路径是否指向文件。

`os.path.isdir`：判断路径是否指向目录。路径结尾的斜杠不会影响结果。

`os.path.join`：最常用的函数之一，能将多个路径连接在一起，自动在每个路径之间依据 `os.sep` 的值添加分隔符。

```Python
# Linux下
In : os.path.join('a', 'b', 'c')
Out: 'a/b/c'

# Windows下
In : os.path.join('a', 'b', 'c')
Out: 'a\\b\\c'
```

这个函数的行为有点复杂，下面再举几个例子

```Python
# Windows下
# 路径中的正斜杠替换掉了os.sep
In : os.path.join('a', 'b/', 'c')
Out: 'a\\b/c'
# 结尾的斜杠会被保留
In : os.path.join('a', 'b', 'c/')
Out: 'a\\b\\c/'
# 最后一个路径为空字符串时,相当于在结尾添加斜杠
In : os.path.join('a', 'b', '')
Out: 'a\\b\\'
```

Linux 下的行为是一样的。另外还有什么路径如果在根目录或盘符下，那么连接时前面的路径会被忽略之类的行为，这里就不细说了。

`os.expanduser`：将一个路径中的 `~` 符号替换成 user 目录的路径。

`os.path` 模块是处理路径的经典模块，但我在使用中遇到的问题是，在 Windows 下如果想使用正斜杠，因为这个模块默认用反斜杠来进行连接和替换操作，会导致产生的字符串中两种斜杠相混杂。虽然这种路径完全合法，但作为结果输出时就很难看。可以考虑使用 `os.path.normpath` 函数来规范化，或者试试下一节将会介绍的模块。

## pathlib

于 Python 3.4 引入的新模块，提供了面向对象风格的路径操作，能够完全替代 `os.path` 和 `glob` 模块，并涵盖一部分 `os` 模块的功能。这里简单介绍一下其用法。

![pathlib](/python_path/pathlib.png)

`pathlib` 中的类由上面的图片表示。最顶层的是 `PurePath`，提供不涉及 I/O 的路径计算；`Path` 类又称 concrete path，继承 `PurePath` 的同时提供 I/O 的功能；剩下的几个类从名字可以看出是与平台相关的，我们一般不需要关心，让程序自动决定即可。

前面提到的路径都是字符串，但 `pathlib` 会把路径作为一个对象

```Python
from pathlib import Path
p = Path('a/b/c')

# Linux下
In : p
Out: PosixPath('a/b/c')
# 获取字符串
In : str(p)
Out: 'a/b/c'

# Windows下
In : p
Out: WindowsPath('a/b/c')
# 获取字符串
In : str(p)
Out: 'a\\b\\c'
```

`Path` 对象内部以正斜杠的形式表示路径，在转换成字符串时会自动根据系统选取分隔符，另外还会自动去掉路径结尾的斜杠。这下我们就不用操心斜杠混用的问题。下面便来介绍 `Path` 对象的方法和属性。需要注意的是，很多方法返回的依然是 `Path` 对象。

`Path.exists`：判断路径是否存在。

`Path.is_file`：判断路径是否指向文件。

`Path.is_dir`：判断路径是否指向目录。

`Path.cwd`：同 `os.getcwd`。

`Path.iterdir`：同 `os.listdir`，不过返回的是生成器。

`Path.mkdir`：创建该路径表示的目录。`parent` 参数指定是否顺带着将不存在的父目录也也一并创建了，等同于 `os.makedirs` 的功能。

`Path.rmdir`：删除该路径表示的空目录。

`Path.touch`：创建该路径表示的文件。

`Path.open`：相当于对路径指向的文件调用 `open` 函数。

`Path.unlink`：删除一个文件或者符号链接。

`Path.rename`：同 `os.rename`。

`Path.replace`：同 `os.replace`。

`Path.resolve`：得到绝对路径，或解析符号链接。

`PurePath.name`：属性，同 `os.path.basename`。

`PurePath.parent`：属性，同 `os.path.dirname`。可以写出 `p.parent.parent` 这样的表达。

`PurePath.parents`：属性，由不同层级的父目录的路径组成的序列。例如 `p.parents[0]` 等于 `p.parent`，`p.parents[1]` 等于 `p.parent.parent`。

`PurePath.suffix`：属性，返回文件的扩展名（含 `.`），如果是目录则返回空字符串。

`PurePath.stem`：属性，返回文件名不含扩展名的那一部分，如果是目录就直接返回目录名。

`PurePath.joinpath`：同 `os.path.join`。不过现在通过重载运算符 `/`，有了更方便的表达

```Python
In : Path('a') / 'b' / 'c'
Out: WindowsPath('a/b/c')
```

`Path.expanduser`：同 `os.path.expanduser`。

`Path.glob`：同 `glob.iglob`，即返回的是生成器。不过现在不需要指定 `recursive` 参数，当模式中含有 `**` 时就会进行递归搜索。

`Path.rglob`：相当于在 `Path.glob` 的模式里提前加上了 `**/`。即 `Path.glob('**/*')` 等同于 `Path.rglob('*')`。

可以看到 `pathlib` 提供了丰富的路径操作，再结合 `shutil` 就足以应对日常使用。另外在 Python 3.6 之后，`os` 与 `os.path` 中许多函数能够直接接受 `Path` 对象作为参数，于是这些模块完全可以互通。`pathlib` 的缺点也不是没有

- Python 3.6 以后才算得上完善，并且 API 以后可能会发生变化。

- 读取文件时多一道将 `Path` 对象转换成字符串的步骤。

## 小结

以上记录了最常用的功能。回到本文开头的问题，我觉得 Windows 平台下可以选择下面的方案

- `os` + `os.path`，路径分隔符全部采用反斜杠。

- `pathlib`，路径分隔符全部采用正斜杠。

到底选哪种，以后慢慢实践就知道了。

## 参考资料

[What is the difference between path and directory?](https://unix.stackexchange.com/questions/131561/what-is-the-difference-between-path-and-directory)

[Windows 的路径中表示文件层级为什么会用反斜杠，而 UNIX 系统都用斜杠？](https://www.zhihu.com/question/19970412/answer/15479052)

[Should a directory path variable end with a trailing slash?](https://stackoverflow.com/questions/980255/should-a-directory-path-variable-end-with-a-trailing-slash)

[Python os 模块详解](https://zhuanlan.zhihu.com/p/150835193)

[How is Pythons glob.glob ordered?](https://stackoverflow.com/questions/6773584/how-is-pythons-glob-glob-ordered)

[你应该使用pathlib替代os.path](https://zhuanlan.zhihu.com/p/87940289)

