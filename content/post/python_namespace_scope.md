---
title: "Python 系列：命名空间和作用域"
date: 2021-10-23
showToc: true
tags:
- python
---

## 定义

### 命名空间

**命名空间（namespace）**：官方说法是从名称到对象的映射，实际上就是保存变量名与变量值绑定关系的一个空间。赋值语句会将绑定关系写入命名空间，而引用变量时则会根据变量名在命名空间中查询出对应的值。并且大部分的命名空间都是利用 Python 的字典来实现的（例外如类的 `__slots__` 属性）。程序中出现在全局的变量构成一个命名空间，Python 内置的函数和异常类也有它们自己的命名空间，每次定义函数或类时也会创建专属于它们的命名空间。命名空间之间相互独立，同名的变量可以存在于不同的命名空间中，例如两个函数内部可以使用同名的局部变量，这有助于我们在不引发冲突的同时合理复用变量名。

### 作用域

**作用域（scope）**：官方说法是 Python 程序中能直接访问一个命名空间的文本区域。听起来有点抽象，实际上就是指出程序中哪些区域的文本归哪个命名空间管理，例如函数的作用域显然就是函数体（定义函数的所有语句），全局作用域就是从头到尾整个程序。但并不是说出现在一个作用域中的变量就一定属于该作用域（的命名空间）：若在该区域内通过赋值语句等操作创建（或修改）了该变量的绑定关系后，那它就属于该作用域；否则它就属于其它作用域，在当前区域引用它需要根据特定的规则向其它作用域进行查询。例如常见的在函数中引用全局变量。本文的一个重点就是要来仔细说说这一规则。

<!--more-->

## LEGB 规则

引用变量时，按 L -> E -> G -> B 的顺序在不同作用域中查询：

- L（Local）：局部作用域，比如函数或方法内部。
- E（Enclosing）：外层作用域，比如一个闭包函数的外层函数部分。
- G（Global）：全局作用域，比如当前运行的文件或导入的模块的内部。
- B（Built-in）：Python 的内置函数等存在的作用域。

举个例子，若在函数中引用某变量，首先会在函数的局部作用域中查询该变量是否存在，查不到就到外层函数（如果存在的话）的作用域里去查，再查不到就接着去全局和内置作用域，如果都查不到就会抛出 `NameError` 异常了。下面再以一张图为例一步步进行解说。

![namespace_scope](/python_namespace_scope/namespace_scope.png)

这段程序的运行结果是

```
func_arg in global: 1
func_arg in func: 2
inner_var in inner_func: 2
outer_var in inner_func: 1
```

首先，程序在启动时就已经全部处于内置作用域中（图中肉色部分）。然后程序的每一句被解释器执行：函数名 `func` 和 `outer_func` 通过 `def` 语句分别绑定给了两个函数对象，其绑定关系写入了全局作用域的命名空间中（图中绿色部分）。`__main__` 是全局作用域中预定义的变量，在本例中值为 `'main'`，变量名 `func_arg` 通过赋值语句绑定给了整数 1。因为全局作用域中并不存在名为 `print` 的函数，所以会到内置作用域中查询，因为 `print` 正好是内置函数所以顺利地找到了——即 G -> B 的查询顺序。`print` 函数的参数中出现了 `func_arg`，全局作用域中就有，所以打印出了整数 1。

接着到了调用函数的部分。我们都知道，函数被调用时会把形式参数（`func_arg`）绑定给传入的实际参数（即整数 2），所以 `func` 的命名空间中出现了 `func_arg`（图中第一个蓝色部分），并且这个 `func_arg` 与全局作用域中的 `func_arg` 毫无干系。然后又是按 L -> G -> B 的顺序在内置作用域中找到 `print` 函数，打印出整数 2。

主程序的最后一句是调用存在嵌套的函数 `outer_func`。`outer_func` 的函数体被执行，其中变量名 `outer_var` 被绑定给整数 1，函数名 `inner_func` 被绑定给嵌套定义的函数对象，之后它们出现在 `outer_func` 的命名空间中（图中第二个蓝色部分）。`outer_func` 函数体的最后一句是调用刚刚定义好的 `inner_func` 函数，`inner_func` 的函数体同样也是一个局部作用域（图中黄色部分），但因为被定义在 `outer_func` 内，所以 `outer_func` 的局部作用域同时也是 `inner_func` 的外部作用域。因此 `inner_func` 中调用 `print` 时发生了 L -> E -> G -> B 的搜索过程。在 `inner_func` 中调用 `outer_var` 也发生了 L -> E 的查询过程。

简单总结一下：作用域就好比花花绿绿的便利贴，最底下两张大的便利贴分别是内置作用域和全局作用域。定义新函数时会在这两张纸的基础上一层一层往上盖小便利贴，因而不同函数栈会摞成一个个纸堆。引用变量时则会从当前便利贴出发，一层一层往下查询，最远查到底层的内置作用域；不过往上查询是不允许的，所以外层函数无法引用内层函数的变量。根据这一规则，不同函数栈之间也是互不相通的。下图是对这一比喻的立体化展示

![query](/python_namespace_scope/query.png)

## nonlocal 和 global 语句

考虑下面这个函数

```python
def outer_func():
    outer_var = 1
    def inner_func():
        outer_var = 2
    print('outer_var before inner_func:', outer_var)
    inner_func()
    print('outer_var after inner_func:', outer_var)
```

运行结果为

```
outer_var before inner_func: 1
outer_var after inner_func: 1
```

明明函数 `inner_func` 对变量 `outer_var` 进行了修改，但修改效果似乎没有体现在外层。这是因为 `outer_var = 2` 这个赋值语句只是在 `inner_func` 的作用域中新定义了一个绑定关系，这里的 `outer_var` 和外层的 `outer_var` 实际上分别属于不同的两个命名空间，除了变量名恰好相同以外并没有任何联系。这一行为还可以解读成，作用域外层的变量总是“只读”的——你可以根据 LEGB 规则引用外层变量的值，但若想通过赋值语句等操作改变其绑定关系，则只会在当前作用域里创建同名变量而已。

若把 `inner_func` 中的赋值语句改为自增

```python
def inner_func():
    outer_var += 1
```

运行却发现会抛出 `UnboundLocalError` 异常。这里自增语句 `outer_var += 1` 等价于赋值语句 `outer_var = outer_var + 1`，我们可能会认为，等号右边会通过引用外层 `outer_var` 的值计算出整数 2，然后再在当前作用域中创建同名的绑定关系，程序应该能正常运行才对。但实际情况是，函数在被定义时，若函数体内存在关于某变量的绑定语句，那么这个变量就一定会被解析到函数自己的作用域中，不会再向外查询——哪怕函数还没被调用、该语句还没被执行。所以当 `inner_func` 看到自己的语句块中出现了自增语句时，就认定 `outer_var` 肯定是自己的局部变量（local），但真当运行到 `outer_var + 1` 的表达式时，却发现局部作用域中查不到它，所以自然产生了 `UnboundLocalError` 异常：该局部变量还没有绑定关系就被引用了，命名空间里查不到它啊。

如果真想修改外部作用域里的绑定关系，就需要用 `nonlocal` 和 `global` 语句显式声明某变量所处的作用域，同时获得修改其绑定关系的权限。`nonlocal` 会把变量名解析到离当前局部作用域最近的非全局的外层作用域中，例如上面的 `inner_func` 可以修改为

```python
def inner_func():
    nonlocal outer_var
    outer_var = 'abc'
```

运行结果为

```
outer_var before inner_func: 1
outer_var after inner_func: abc
```

可以看到通过 `nonlocal` 声明 `inner_func` 里的 `outer_var` 就是外层那个 `outer_var`，便可以在 `inner_var` 里修改 `outer_var` 的绑定关系。`global` 同理，不过顾名思义会把变量名解析到全局作用域，例如

```python
N = 10
def func():
    global N
    N += 10

if __name__ == '__main__':
    print('N before func:', N)
    func()
    print('N after func:', N)
```

运行结果为

```
N before func: 10
N after func: 20
```

如果去掉 `global` 的语句的话，同样会抛出 `UnboundLocalError` 异常。

需要注意，这一节针对的都是不可变（immutable）对象，若外层作用域的变量是可变（mutable）对象，例如列表、字典等，那么即便不用 `nonlocal` 和 `global` 语句，我们也能用赋值语句直接修改其元素，利用自增语句进行原地的连接操作。

## 模块的作用域

每个模块都有其专属的命名空间和全局作用域，模块内变量的引用同样服从 LEGB 规则。事实上，主程序也不过是特殊的 `__main__` 模块的一部分而已。通过 `import` 语句可以把主程序里的变量名绑定给其它模块里的对象，以实现跨模块的引用。例如

```python
import math
from math import sqrt
```

第一句会将 `math` 模块作为一个对象绑定到主程序里的 `math` 变量名上，接着以 `math.func` 的形式调用模块里的函数即可。而第二句等价于

```python
import math as _
sqrt = _.sqrt
del(_)
```

相当于把 `math.sqrt` 函数直接绑定到主程序里的 `sqrt` 变量名上。因此可以想到，直接修改 `sqrt` 的绑定关系并不会影响到 `math.sqrt`。下面还是再以图片为例

![module](/python_namespace_scope/module.png)

内置作用域上有两个全局作用域（图中绿色部分），左边是主程序的，而右边是自定义的 `mod` 模块的。本来这两个作用域互相独立，但通过 `from mod import exp` 语句将右边的 `exp` 函数导入到了左边，所以现在左边也能调用 `exp`。注意，虽然现在 `exp` 属于主程序的全局作用域，但 `exp` 指向的函数对象直接定义在 `mod.py` 文件中，其内部的变量依然工作在 `mod` 模块的全局作用域里（例如函数中用到了定义在 `mod` 里的全局变量 `e`，不会说导入到主程序中就找不到 `e` 了）。

## 类的作用域

类的说明要稍微麻烦些，所以这里直接通过例子来展示

![class](/python_namespace_scope/class.png)

运行结果为

```
Kate : meow
```

首先，类只有当其定义里的语句被全部执行后才能生效（显然函数不是这样）。当程序刚进入类定义时会创建类专属的命名空间，之后定义里的绑定关系将会被记录到这个命名空间中。如图中蓝色部分所示，绑定了一个类变量 `sound` 和两个类函数 `__init__` 和 `call`，同时这两个函数因为第一个参数是 `self`，所以之后还能作为实例的方法被调用。定义执行完毕后会创建一个类对象，并将其绑定到与类名同名的名称上去（此处是 `Cat`）。

直接调用类对象可以创建一个空的实例对象 `c`，它也有自己独立的命名空间。我们可以通过 `c.attr` 的形式引用类相关的变量。若引用的是实例变量，那么会直接查询实例自己的命名空间；若引用的是类变量，那么会跳到实例所属的类的命名空间中去查找；若引用的是方法，则会跳到实例所属的类的命名空间中查找同名的函数，并将实例对象自身作为 `self` 参数传入。

再回过头来看具体的程序，`Cat` 类在被直接调用时会自动调用 `__init__` 方法（如果存在的话），同时将 `c` 和接收的其它参数一并传给 `__init__`。`__init__` 的作用是给实例一个初始状态，可以看到函数定义里以 `self.name = name` 等赋值语句向 `c` 的命名空间中写入了实例变量的绑定关系。之后主程序中调用 `c.call()`，等价于 `Cat.call(c)`，`call` 的函数定义中 `self.sound` 又等价于 `Cat.sound`。

类与函数的一个重要差别是，函数里嵌套定义的函数可以按 L -> E 的顺序引用外层函数的变量，但类里定义的函数并不能引用类变量，例如本例中 `call` 函数里直接引用 `sound` 会抛出 `NameError` 异常。虽然类也有命名空间和作用域，但内层函数在向外层查询时会跳过类的作用域，用图上的内容来说，就是蓝色层对于黄色层是“透明”的。不过，因为绿色层里有类对象的绑定关系，所以可以用 `Cat.attr` 的形式迂回引用类属性。

总结一下：类的作用域不同于一般函数的作用域，类里的函数不能直接访问类属性，但可以委托实例对象（`self`）去访问类变量和类方法，或直接用类名访问所有类属性。如果存在继承，那么上面提到的委托操作会递归地向父类进行查询，这里篇幅有限就不再详谈了。

PS：如果你尝试以下代码

```python
N = 10
print('N before class:', N)
class A:
    N += 10
    print('N in class:', N)
print('N after class:', N)
```

运行结果为

```
N before class: 10
N in class: 20
N after class: 10
```

Emmm……对于函数会报 `UnboundLocalError` 错误，但对类就成功运行了。所以也有人说其实类只有命名空间而没有作用域，感兴趣的读者可以参考最后一个参考链接。

## 参考链接

[The Python Tutorial: 9. Classes](https://docs.python.org/3/tutorial/classes.html)

[Python3 命名空间和作用域](https://www.runoob.com/python3/python3-namespace-scope.html)

[Python是一种纯粹的语言](https://zhuanlan.zhihu.com/p/23926957)

[Python的类定义有没有建立新的作用域？](https://www.zhihu.com/question/50688142)
