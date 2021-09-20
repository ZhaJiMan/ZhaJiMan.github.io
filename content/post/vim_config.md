---
title: "简单的 vim 配置"
date: 2021-09-20
showToc: true
tags:
- vim
---

最近越发老年痴呆，连自己写的 vim 配置的作用都忘光了，所以在本文记录并解说一下我常用的配置以便查阅。这里的配置非常简单，仅用以强化基本的使用体验。因为我工作在内网服务器上，所以也不涉及联网部分（例如 [vim-plug](https://github.com/junegunn/vim-plug) 插件）。文中 vim 版本是 7.4。

<!--more-->

## 具体配置

首先关闭对 vi 的兼容，并保证退格键能正常使用

```
" 关闭对vi的兼容
set nocompatible
" 设置backspace键功能
set backspace=eol,start,indent
```

设置行的显示

```
" 显示行号
set number
" 高亮显示当前行
set cursorline
" 让一行的内容不换行
set nowrap
" 距窗口边缘还有多少行时滚动窗口
set scrolloff=3

" 显示标尺,提示一行代码不要超过80个字符
set ruler
set colorcolumn=80
```

设置缩进。Vim 默认使用宽度为 8 的 tab，而我一般写 Python，需要用 4 个空格替代 tab。这里参考 [Useful VIM Settings for working with Python](http://www.vex.net/~x/python_and_vim.html) 的设置。关于这些选项的意义可以参考 vim 的帮助文档或 [Secrets of tabs in vim](https://tedlogan.com/techblog3.html)

```
" tab设为4个空格
set tabstop=4
set shiftwidth=4
set softtabstop=4
set expandtab
set smarttab
" 新一行与上一行的缩进一致
set autoindent
```

显示相匹配的括号，并增强搜索功能。搜索产生的高亮可以通过 `:nohlsearch` 命令消除掉。

```
" 显示括号匹配
set showmatch
" 高亮查找匹配
set hlsearch
" 增量式搜索
set incsearch
" 不区分大小写,除非含有大写字母
set ignorecase
set smartcase
```

开启语法高亮并设置配色。这里使用的是 [onedark](https://github.com/joshdick/onedark.vim) 配色方案

```
" 开启语法高亮
syntax on
" 代码颜色主题
set t_Co=256
colorscheme onedark
```

增强命令部分的显示和补全，并开启状态栏（状态栏可以用 [vim-airline](https://github.com/vim-airline/vim-airline) 插件替代）

```
" 在右下角显示部分命令
set showcmd
" 命令可以用tab补全,并设置匹配规则
set wildmenu
set wildmode=list:longest,full
" 总是显示状态栏
set laststatus=2
```

显示 tab 和行尾多余的字符

```
" 显示tab和行尾多余的空格
set list
set listchars=tab:>·,trail:·
```

Vim 中通过 `:vsp file` 命令在水平方向上打开一个新文件，`:sp file` 则是垂直方向。但它们默认打开的位置分别是左边和上边，这里按我的习惯改成在右边和下边打开。另外在分屏间移动光标的命令是 `<C-w> + hjkl`，移动分屏的命令是 `<C-w> + HJKL`，`<C-w>` 需要我使劲扭曲左手才能按到，非常费劲，故这里用轻松好按的 `<space>` 代替

```
" 设置分屏时的位置
set splitright
set splitbelow
" 设置在分屏间移动的快捷键
map <space>h <C-w>h
map <space>l <C-w>l
map <space>j <C-w>j
map <space>k <C-w>k
" 设置移动分屏的快捷键
map <space>H <C-w>H
map <space>L <C-w>L
map <space>J <C-w>J
map <space>K <C-w>K
```

检测文件类型、设置 vim 内部的字符编码为 utf-8，对文件的解码参考 [用vim打开后中文乱码怎么办？](https://www.zhihu.com/question/22363620) 中马宏菩的回答，防止中文出现乱码

```
" 检测文件类型
filetype on
" 文件编码
set encoding=utf-8
set fileencodings=ucs-bom,utf-8,utf-16,gbk,big5,gb18030,latin1
" 没有保存或文件只读时弹出确认
set confirm
```

设置历史记录条数，并禁用自动备份（理由我忘了……）

```
" 记录历史记录的条数
set history=1000
set undolevels=1000
" 禁用自动备份
set nobackup
set nowritebackup
set noswapfile
```

偶尔要用到 NCL 语言，需要相关的高亮提示，所以在 [NCL: Editor enhancements for use with NCL scripts](https://www.ncl.ucar.edu/Applications/editor.shtml) 网址下载 `ncl3.vim` 文件并重命名为 `ncl.vim`，把文件放入 `~/.vim/syntax` 目录中，再修改 `.vimrc` 的配置。其中还为 [vim-commentary](https://github.com/tpope/vim-commentary) 插件设置了 NCL 的注释

```
" NCL高亮设置
au BufRead,BufNewFile *.ncl set filetype=ncl
au! Syntax newlang source $VIM/ncl.vim
" NCL注释设置
autocmd FileType ncl setlocal commentstring=;%s
```

非联网状态下使用 [vim-pathogen]() 插件管理插件目录，在配置里加上这么一句

```
" pathogen设置,用于管理插件
execute pathogen#infect()
```

我目前用到的配置就以上这些。可以说是很简陋了，连自动补全都没有（我用的默认的 `<C-n>`），同时因为版本只有 7，也不好整 REPL。不过我觉得作为基本的文本编辑器已经够用了，如果读者有心得也可以传授我一下。

## 参考链接

[Vim 配置入门](https://www.ruanyifeng.com/blog/2018/09/vimrc.html)

[有哪些编程必备的 Vim 配置？](https://www.zhihu.com/question/19989337)

[上古神器Vim：从恶言相向到爱不释手 - 终极Vim教程01](https://www.bilibili.com/video/BV164411P7tw)
