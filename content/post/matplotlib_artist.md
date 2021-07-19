---
title: 'Matplotlib 中的“Artist”——你在网上浪费时间乱搜一通之前应该知道的东西'
date: 2021-07-14
draft: true
showToc: true
---

> 这是我在 Qiita 上看到的一篇介绍 Artist 的文章（[原文链接](https://qiita.com/skotaro/items/08dc0b8c5704c94eafb9)），觉得对于初学 Matplotlib 的同学来说应该会很有帮助，现将其英文版翻译如下。作者是 @skotaro，写于 2018-03-14。

毫无疑问，Python 中的 matplotlib 是一个非常棒的可视化工具，但在 matplotlib 中调整作图细节也是一件很折磨人的事。你很可能会花上好几个小时来研究怎么修改图里的某一部分，有时你甚至都不知道这部分叫什么，连怎么上网搜都整不明白。就算你在 Stack Overflow 上找到了线索，也可能还要再花几个小时才能把它改到符合自己的需求。其实，只要你去了解一下 matplotlib 里面图的组成以及能对它们做什么，就完全可以避免上述的徒劳。跟你们中的大多数人一样，我也是靠着 Stack Overflow 上各路 matplotlib 高手们写的那一大堆回答才搞定自己的作图问题的。不过最近我发现 [官网的 `Artist` 对象教程](https://matplotlib.org/stable/tutorials/intermediate/artists.html) 写得很有启发性，有助于我们理解 matplotlib 作图时发生了什么，并节省大把调整的时间。在这篇文章中我想分享一些关于 matplotlib 中 `Artist` 对象的基本知识，学了这些说不定就能免去瞎捣鼓的时间。

## 本文的目的

我并不打算写些具体的教程，例如“想要 XX 效果的话你需要 XX”，而是想介绍 matplotlib 中 `Artist` 的基本概念，帮助你挑选合适的搜索关键词，并为遇到的相似的问题

