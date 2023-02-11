---
title: "Python 系列：import 语句"
date: 2022-11-30
draft: true
showToc: true
tags:
- python
---

<!--more-->

Q：

- 什么是模块
- 执行的脚本和主模块
- 什么是包
- `import` 语句怎么找到模块或包
- 包是如何找到自己的子模块/包的
- `__name__`、`__file__` 和 `__dict__` 属性
- 为什么导入要放到脚本最上面
- 循环导入
- 自己写包怎么组织，怎么找到
- `site` 是什么，第三方包和标准库的区别

查看所有第三方包：

- `pip list`
- `conda list`

## 命令行参数

### -c

https://docs.python.org/3/using/cmdline.html

`python -c command` 时，当前目录的相对路径（空字符串）会被加到 `sys.path` 的开头，这样 `command` 里可以引用当前目录的模块。不过 `-c` 选项也没什么用就是了。

### -m

`python -m modname` 会在 `sys.path` 里搜索名为 `modname` 的模块，找到作为 `__main__` 模块执行。`modname` 可以是模块文件，也可以是 package，相当于执行 `modname/__main__.py` 文件。当前目录的绝对路径会被加到 `sys.path` 的开头。`-I` 选项能够禁用当前目录、`site-package` 目录和 `PYTHON*` 环境变量里的目录。

如果模块或 package 在更深的目录层级里，`-m` 选项就会找不到它们，所以必须在和模块同级的目录下执行。

关于 `-m` 的说明：

Note: This option cannot be used with built-in modules and extension modules written in C, since they do not have Python module files. However, it can still be used for precompiled modules, even if the original source file is not available.

例如 `timeit` 这样的标准库模块就能以 `-m` 选项作为脚本执行，提供一些命令行功能。

### 接文件名

`python <script>` 时，`<script>` 必须是一个指向 Python 文件、含 `__main__.py` 文件的目录，或 zip 压缩包的路径（绝对路径或相对路径）。

- 当 `<script>` 是 Python 文件时，文件所在的目录的绝对路径会加到 `sys.path` 的开头。
- 当 `<script>` 是目录或 zip 文件时，名为 `<script>` 的目录的绝对路径会加到 `sys.path` 的开头。

同样可以使用 `-I` 选项隔离这些路径。

如果什么选项也不给，相当于 `python -i`，即进入交互模式。当前目录的相对路径，即空字符串会加到 `sys.path` 的开头。奇怪的是 IPython 是把空字符串加到 `sys.path` 的中间。

### 总结

`python <script>`：将 `<script>` 所在的目录，或以 `<script>` 为名的目录的绝对路径加到 `sys.path` 的开头。

`python -c command`：将当前目录的相对路径（空字符串）加到 `sys.path` 的开头。

`python -m modename`：将当前目录的绝对路径加到 `sys.path` 的开头。

`python` 将当前目录的相对路径（空字符串）加到 `sys.path` 的开头；IPython 加到中间。

## builtin

https://docs.python.org/3/library/builtins.html

内置的函数和常量都放在 `builtin` 模块里，但一般我们不需要导入这个模块，直接在全局就可以调用其内容。大多数模块的全局都有 `__builtins__` 变量，其值通常为 `builtin` 模块或其 `__dict__` 属性。这是一个实现细节，一般不用关心。

## sys.path 的初始化

https://docs.python.org/3/library/sys_path_init.html

首先，如果 `python` 命令后面有文件，那么 `sys.path` 的第一条就是文件所在目录的路径；如果没有，即交互模式、`-c` 和 `-m`，那么第一条是当前目录（空字符串）。

接下来是 `PYTHONPATH` 里的路径。注意 `PYTHONPATH` 会影响到系统上所有 Python 版本和环境，所以需要小心使用。

我的打印结果是：

```
D:\code\python\test
D:\code\github\frykit
D:\conda\python39.zip
D:\conda\DLLs
D:\conda\lib
D:\conda
D:\conda\lib\site-packages
D:\conda\lib\site-packages\certifi-2022.6.15-py3.9.egg
D:\conda\lib\site-packages\win32
D:\conda\lib\site-packages\win32\lib
D:\conda\lib\site-packages\Pythonwin
```

第一行是文件所在目录，第二行是 `PYTHONPATH` 的内容。

接下来是标准模块所在的目录，以及这些模块依赖的扩展模块（用 C 或 C++ 写的，`.pyd` 或 `.so` 文件）所在的目录。`sys.prefix` 是与平台无关的 Python 模块的目录前缀，`sys.exec_prefix` 是扩展模块的目录前缀，不过 `sys` 的文档也说这是与平台相关的 Python 模块的目录前缀。

## Modules

https://docs.python.org/3/tutorial/modules.html

把函数和类定义写在文件里，之后在脚本或解释器中使用其中定义的函数，这种文件即 module。

module 内的定义可以被导入其它 module，或 main module——在最顶层或计算器模式下运行的脚本里可获取的变量的组合。

所谓模块就是一个含有 Python 定义和语句的文件，文件名即模块名，后缀是 `.py`。模块的名字在模块内以全局变量 `__name__` 的形式出现。

### 更多解释

> 下面这段不懂，为什么只在第一次运行时才执行呢？

模块内的语句和函数定义用来初始化模块，它们只在模块的名字第一次被 `import` 语句引用时才会执行。（事实上函数定义也是被执行的语句，效果是将函数名添加到模块的全局命名空间中。）

每个模块都有其私有的命名空间，对于定义其中的函数来说就是全局命名空间。因此，模块的作者可以在模块内随便用全局变量，而不用担心与用户的全局变量发生冲突。模块的全局变量以 `modname.itemname` 的形式获取。

模块里可以导入其它模块。如果把模块放到模块或脚本的最顶上，那么导入的模块名字会被添加到模块的全局命名空间里。（注：换句话说函数里导入的会添加到局部命名空间里？）

`from modname import itemname` 的形式不会在当前局部命名空间里引入 `modname`。而 `from modname import *` 会把导入模块里所有不以 `_` 开头的名字。当然一般不推荐这样导入。`as` 用来将 `import` 后面的对象绑定给 `as` 后面的名字。

出于效率上的考虑，在解释器的会话里每个模块只会导入一次，因此如果模块的内容改变了，需要重启解释器，或者调用 `importlib` 模块的 `reload` 方法。

#### 模块作为脚本执行

```Python
python modname.py <arguments>
```

模块被执行，其 `__name__` 的值为 `'__main__'`。所以可以在模块最后添加

```Python
if __name__ == '__main__':
    ...
```

这段代码只会在脚本作为“主文件”时才会被执行（例如放测试语句）。当文件作为模块被导入时，这段代码就不会被执行。

#### 模块搜索路径

解释器搜索名为 `spam` 的模块时，首先会找这个名字的内置（built-in）模块，名字列在 `sys.builtin_module_names` 里。如果没找着，就会在 `sys.path` 里列出来的目录里找 `spam.py`。`sys.path` 的内容为：

- 输入脚本的所在目录的绝对路径，并且排第一位。
- `PYTHONPATH`，系统环境变量里的一串目录名。
- 跟安装相关的，例如 `site-packages` 目录。

`sys.path` 初始化后，Python 程序可以修改其内容。其中脚本所在目录的路径比标准库路径靠前，因而同名时会优先在脚本目录找。

> https://docs.python.org/3/library/sys_path_init.html：搜索模块的路径在 Python 启动时初始化，通过 `sys.path` 获取。

## Python 标准库

https://docs.python.org/zh-cn/3/library/index.html

> Python 标准库非常庞大，所提供的组件涉及范围十分广泛，正如以下内容目录所显示的。这个库包含了多个内置模块 (以 C 编写)，Python 程序员必须依靠它们来实现系统级功能，例如文件 I/O，此外还有大量以 Python 编写的模块，提供了日常编程中许多问题的标准解决方案。其中有些模块经过专门设计，通过将特定平台功能抽象化为平台中立的 API 来鼓励和加强 Python 程序的可移植性。

划重点：标准库包含内置模块（C 写的）。

官方文档是从内置函数、常量、类型和异常开始讲起，接着介绍需要 `import` 的包/模块。

## sysconfig

`sysconfig` 模块可以查询安装路径，安装方案（scheme）会因平台和安装选项的不同而不同。九种方案为：

- `posix_prefix`
- `posix_home`
- `posix_user`
- `posix_venv`
- `nt`
- `nt_user`
- `nt_venv`
- `venv`
- `osx_framework_user`

posix 指 Linux 或 macOS，nt 指 Windows。这里不打算深究方案的含义。每个方案由 8 个路径组成：

- `stdlib`: directory containing the standard Python library files that are not platform-specific.
- `platstdlib`: directory containing the standard Python library files that are platform-specific.
- `platlib`: directory for site-specific, platform-specific files.
- `purelib`: directory for site-specific, non-platform-specific files.
- `include`: directory for non-platform-specific header files for the Python C-API.
- `platinclude`: directory for platform-specific header files for the Python C-API.
- `scripts`: directory for script files.
- `data`: directory for data files.

用 `sysconfig.get_paths` 以字典的形式查看当前平台默认方案的所有路径（目录），`get_path` 可以指定上面的 8 个名字。以 Linux 平台 conda 的 base 和 test 环境为例：

```Python
{'stdlib': '/data/anaconda3/lib/python3.9',
 'platstdlib': '/data/anaconda3/lib/python3.9',
 'purelib': '/data/anaconda3/lib/python3.9/site-packages',
 'platlib': '/data/anaconda3/lib/python3.9/site-packages',
 'include': '/data/anaconda3/include/python3.9',
 'platinclude': '/data/anaconda3/include/python3.9',
 'scripts': '/data/anaconda3/bin',
 'data': '/data/anaconda3'}
```

```Python
{'stdlib': '/data/anaconda3/envs/test/lib/python3.10',
 'platstdlib': '/data/anaconda3/envs/test/lib/python3.10',
 'purelib': '/data/anaconda3/envs/test/lib/python3.10/site-packages',
 'platlib': '/data/anaconda3/envs/test/lib/python3.10/site-packages',
 'include': '/data/anaconda3/envs/test/include/python3.10',
 'platinclude': '/data/anaconda3/envs/test/include/python3.10',
 'scripts': '/data/anaconda3/envs/test/bin',
 'data': '/data/anaconda3/envs/test'}
```

再以 Windows 平台的 base 为例：

```Python
{'stdlib': 'D:\\conda\\Lib',
 'platstdlib': 'D:\\conda\\Lib',
 'purelib': 'D:\\conda\\Lib\\site-packages',
 'platlib': 'D:\\conda\\Lib\\site-packages',
 'include': 'D:\\conda\\Include',
 'platinclude': 'D:\\conda\\Include',
 'scripts': 'D:\\conda\\Scripts',
 'data': 'D:\\conda'}
```

可以总结出的信息是，base 环境的根目录是 `/data/anaconda3`，而 test 环境在其下的 `envs/test` 里，二者子路径的结构相同。标准库都在 `lib` 目录里，第三方包都是在 `site-packages` 里，头文件在 `include` 里，脚本在 `bin` 或 `Scripts` Linux 和 Windows 的细节有很多不同，原因不明。

## pass

https://www.zhihu.com/question/30296617/answer/112564303

虚拟机执行脚本的过程：

- 完成模块的加载和链接。
- 将源代码翻译为 `PyCodeObject` 对象（即字节码），并将其写入内存中。
- 从内存里读取指令并执行。
- 程序结束后根据命令行调用情况，决定是否将 `PyCodeObject` 写回硬盘当中（即 `.pyc` 或 `.pyo` 文件）。
- 之后若再次执行该脚本，则先检查本地是否有上述字节码文件。

模块每次导入前都会检查字节码文件的修改时间是否和自身一致，否的话重新生成字节码并覆盖原文件。

`.pyc` 文件的加载速度比 `.py` 有所提高，并且可以隐藏源码，起一定程度的反编译作用。`.pyo` 是优化 `.pyc` 后生成的更小的文件。

## Python 是如何检索包路径的

https://zhuanlan.zhihu.com/p/426672057

`D:\\conda\\lib\\site-packages` 貌似是 Conda 的第三方包位置。

`site` 模块和 `.pth` 文件？在 `site.getsitepackages()` 里的路径里，搜索 `.pth` 文件，将其中的路径载入 `sys.path`。