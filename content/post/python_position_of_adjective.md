---
title: "变量命名时形容词应该放在名词前面还是后面？"
date: 2022-01-05
shotToc: false
tags:
- python
---

今天改程序时脑海里突然蹦出这个问题，更宽泛地说，是修饰词或者偏正结构的先后顺序，例如

- `upper_ax` 和 `bottom_ax`，`ax_upper` 和 `ax_bottom`。
- `start_date` 和 `end_date`，`date_start` 和 `date_end`。

一旦开始疑惑，焦虑便随之而来：哪一种比较好呢？我之前的代码里好像两种写法都出现过，有没有什么现成的规范可以参考呢？越想越不痛快，所以赶紧上网找点前人经验来背书。意外的是，网上大部分文章都在讨论如何取有意义的变量名，而关于这个问题的寥寥无几，也许是因为太细节、太“语法”了？现归纳两篇我看过的帖子以供参考。

首先在 stack overflow 上找到了[一模一样的提问](https://stackoverflow.com/questions/36504357/should-variable-names-have-adjectives-before-or-after-the-noun)：是用 `left_button` 和 `right_button`，还是 `button_left` 和 `button_right` 更好呢？提问者自己觉得前者符合英文语序，读起来更加自然，而后者强调了变量的重点在于按钮，而左和右是额外的补充信息。有评论指出后者在 IDE 里更方便，因为你一键入 `button`，就会自动联想出所有带后缀的版本。这也挺符合人的联想过程，我们肯定是先想到“我要找按钮”，再明确具体要什么样的按钮。当然也有评论给出了经典的废话：与其纠结哪一种约定，任选一种并在项目里维持一致性最重要！好家伙，要是我如此豁达还会来搜这种鸡毛蒜皮的问题吗？

<!--more-->

接着我在 reddit 上找到了 [相关的讨论串](https://www.reddit.com/r/Python/comments/82wlsx/variable_naming_do_you_put_the_adjective_before/)，里面有两个很有说服力的回复，稍微总结如下。

网友 DarkSilkyNightmare 表示根据 ta 多年的经验，回答是：”**永远把最大的单元放在前面**。“例如在面向对象的 Python 里，所谓最大的单元通常是某个类。再具体点，如果想表示一个 `file` 有很多属性，那就采用 `file_id` 和 `file_name` 这种命名；如果想表示存在很多 `id`，并强调 `file` 的 `id` 不同于 `thread` 和 `program` 的，那就采用 `id_thread` 和 `id_file` 这种命名。类似地，如果你有一个含许多属性的 `earth`，就用 `earth_radius`；如果有一系列天体的半径，那么 `radius_earth` 和 `radius_sun` 之类的名字会更有意义。

遵循这个规则可不是迂腐，反而会带来很多具体的好处。首先，这样命名可以提高代码的可读性，让对象所描述的内容更加清晰，并暗示程序的结构。比如说，看到 `radius_earth` 就会联想到可能存在其它 `raidius_*` 形式的变量，而看到 `earth_radius` 就会联想到其它 `earth_*` 形式表示地球属性的变量。其次，这可以让代码更容易归档和搜索。例如在代码文档工具中相似的变量会按字母顺序排在一起，显然形容词在前时会把顺序搞得一团糟。最后，这有助于我们使用工具扩展代码，例如想把 `earth_radius` 中的 `earth` 改成一个类，那么批量修改变量名时将 `earth_` 替换成 `earth.` 即可。

另一位网友 t_h_r_o_w_-_a_w_a_y 也给出了很深刻的见解。变量命名应该做到便于人类阅读和理解，而英语里形容词一般前置，所以很多人认为变量命名时这样做最为合理。但问题在于，合理与否的标准不应该是”因为某门语言就是这样规定的“，而应该是”大部分读者就是这样感觉的“。事实上，程序员并不都来自英语国家，只是当下他们必须用英语编程罢了，世界上约有一半的语言形容词经常后置，例如法语、西班牙语、葡萄牙语、意大利语等（详见这篇 [知乎回答](https://www.zhihu.com/question/394291089/answer/1252033863)）。考虑到读者来源的多样性，不能断言形容词后置就是不合规范的，兴许别人读起来更习惯呢。再激进点说，按前文提到的人的正常联想过程，恐怕英语才是真正不合理的那位吧！

抛开这些主观意见，形容词后置的视觉效果是明显优于前置的。例如定义一系列物体的半径

```python
earth_radius = 243
io_radius = 12
pluto_radius = 30
your_mom_radius = 100000
gliese_581g_radius = 500
```

```python
radius_earth = 243
radius_io = 12
radius_pluto = 30
radius_your_mom = 100000
radius_gliese_581g = 500
```

哪种更清晰一目了然。

本来只是想搜搜相关的规范，结果看到了从实用性和语言学角度的解释，算是意外之喜。最后列出一般的命名规范以供参考：

- [PEP 423 -- Naming conventions and recipes related to packaging](https://www.python.org/dev/peps/pep-0423/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)（中文版 [Python风格规范](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)）
- [What is the naming convention in Python for variable and function names?](https://stackoverflow.com/questions/159720/what-is-the-naming-convention-in-python-for-variable-and-function-names)
