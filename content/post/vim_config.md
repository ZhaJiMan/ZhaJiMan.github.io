---
title: "简单的 Vim 配置"
date: 2021-09-20
showToc: true
tags:
- vim
---

最近越发老年痴呆，连自己写的 Vim 配置的作用都忘光了，所以在本文记录并解说一下我常用的配置以便查阅。这里的配置非常简单，仅用以强化基本的使用体验。由于我同时工作在能联网的 PC 和内网的服务器上，所以也会分开介绍如何在这两种环境下安装插件。文中 Vim 版本分别是 8.1（PC）和 7.4（服务器）。

![vim](/vim_config/vim.png)

<!--more-->

## 基本配置

首先介绍 Vim 自带的基本配置，配置文件的路径是 `~/.vim/vimrc`。关闭对 vi 的兼容，并保证退格键能正常使用

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
set scrolloff=8

" 显示标尺,提示一行代码不要超过80个字符
set ruler
set colorcolumn=80
```

设置缩进。Vim 默认使用宽度为 8 的 tab，而我一般写 Python，需要用 4 个空格替代 tab。这里参考 [Useful VIM Settings for working with Python](http://www.vex.net/~x/python_and_vim.html) 的设置。关于这些选项的意义可以参考 Vim 的帮助文档或 [Secrets of tabs in vim](https://tedlogan.com/techblog3.html)

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

显示相匹配的括号，并增强搜索功能。

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

开启语法高亮并设置配色。这里使用的是 [onedark.vim](https://github.com/joshdick/onedark.vim) 配色方案（后面会介绍如何安装）

```
" 开启语法高亮
syntax on
" 代码颜色主题
set t_Co=256
colorscheme onedark
```

增强命令部分的显示和补全

```
" 在右下角显示部分命令
set showcmd
" 命令可以用tab补全,并设置匹配规则
set wildmenu
set wildmode=list:longest,full
```

显示 tab 和行尾多余的字符

```
" 显示tab和行尾多余的空格
set list
set listchars=tab:>·,trail:·
```

切换 buffer 时 Vim 总会提醒你将当前 buffer 的改动写入文件。打开 `hidden` 能允许我们将未保存的 buffer 放到后台。水平分屏和垂直分屏操作分别默认在上边和左边打开一个新 window，这不太符合我的习惯，所以改为在下边和右边创建

```
" 允许隐藏未保存的buffer
set hidden
" 设置分屏时的位置
set splitright
set splitbelow
```

检测文件类型、设置 Vim 内部的字符编码为 utf-8，对文件的解码参考 [用vim打开后中文乱码怎么办？](https://www.zhihu.com/question/22363620) 中马宏菩的回答，防止中文出现乱码

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

偶尔要用到 NCL 语言，需要相关的高亮提示，所以在 [NCL: Editor enhancements for use with NCL scripts](https://www.ncl.ucar.edu/Applications/editor.shtml) 网址下载 `ncl3.vim` 文件并重命名为 `ncl.vim`，把文件放入 `~/.vim/syntax` 目录中，再修改 `.vimrc` 的配置。其中还为 [commentary.vim](https://github.com/tpope/vim-commentary) 插件设置了 NCL 的注释

```
" NCL高亮设置
au BufRead,BufNewFile *.ncl set filetype=ncl
au! Syntax newlang source $VIM/ncl.vim
" NCL注释设置
autocmd FileType ncl setlocal commentstring=;%s
```

## 插件配置

### PC

PC 上使用 [vim-plug](https://github.com/junegunn/vim-plug) 插件来管理其它插件，利用它能非常简单地安装、更新和移除插件，关于它的安装和使用方法详见其 GitHub 页面。在 `vimrc` 文件的开头添加

```
call plug#begin('~/.vim/plugged')
Plug 'joshdick/onedark.vim'
Plug 'vim-airline/vim-airline'
Plug 'tpope/vim-commentary'
Plug 'kshenoy/vim-signature'
Plug 'mhinz/vim-startify'
Plug 'junegunn/fzf'
Plug 'junegunn/fzf.vim'
call plug#end()
```

保存后再执行命令 `PlugInstall` 即可将 `call` 语句块中提到的插件下载并安装到 `~/.vim/plugged` 目录下。这里用到的插件有：

- [onedark.vim](https://github.com/joshdick/onedark.vim)：一个暗配色方案。
- [vim-airline](https://github.com/vim-airline/vim-airline)：更好看的状态栏。
- [commentary.vim](https://github.com/tpope/vim-commentary)：引入注释命令。
- [vim-signature](https://github.com/kshenoy/vim-signature)：显示出 mark 标记。
- [vim-startify](https://github.com/vim-scripts/vim-startify)：给 vim 整个开屏页面。
- [fzf](https://github.com/junegunn/fzf) 和 [fzf.vim](https://github.com/junegunn/fzf.vim)：引入模糊搜索功能。

如果总是下载失败，可以考虑给 Git 设置代理。

每个插件都可以再进行单独配置，这里我只改动了 vim-startify：在 Vim 的启动界面显示最近打开过的 15 个文件，并添加 `~/.bashrc` 和 `~/.vim/vimrc` 两个文件到收藏夹

```
" vim-startify的设置
let g:startify_files_number = 15
let g:startify_lists = [
    \ {'type': 'files', 'header': ['   Recent Files']},
    \ {'type': 'bookmarks', 'header': ['   Bookmarks']}
    \ ]
let g:startify_bookmarks = [
    \ {'b': '~/.bashrc'},
    \ {'v': '~/.vim/vimrc'}
    \ ]
```

### 服务器

对于不能联网的服务器，依据 vim-plug 作者的建议（[issue #808](https://github.com/junegunn/vim-plug/issues/808)），用 [pathogen.vim](https://github.com/tpope/vim-pathogen) 插件代替 vim-plug。不同于 vim-plug，pathogen.vim 并不能帮你下载插件，它的功能只是将其它插件的路径添加到 Vim 的 `runtimepath` 中，使 Vim 能在工作时找到其它插件罢了。首先在 GitHub 上下载 `pathogen.vim` 文件并移动到服务器的 `~/.vim/autoload` 目录下，再手动下载其它插件的仓库，解压并重命名，移动到服务器的 `~/.vim/bundle` 目录下，最后在 `vimrc` 文件的开头添加

```
" 把插件加入runtimepath
execute pathogen#infect()
```

我们所需的插件即可生效。如果你服务器上的 Vim 版本是 8，那么连 pathogen.vim 也不需要，直接使用原生的 `pack` 语句块即可，我没用过所以就不解说了。

## 快捷键配置

为了不与 normal 模式下已有的大量快捷键发生冲突，所以这里用 `<Leader>` 键作为自定义快捷键的起手式。关于 `<Leader>` 键的解说可见 [How to Use the Vim \<leader\> Key](https://tuckerchapman.com/2018/06/16/how-to-use-the-vim-leader-key/)，这里使用趁手的空格键作为 `<Leader>` 键。自定义快捷键可以解决以下痛点

- 每次搜索产生的高亮需要通过 `:nohlsearch`（或简化的 `:noh`）命令取消，包括回车在内至少要按 5 个键。改成快捷键后就只需要按 2 下。
- 经常要用 fzf.vim 插件的 `:Files` 和 `:Buffers` 命令打开文件，设成快捷键更方便。
- 在分屏中移动光标的的默认快捷键是 `<c-w> + hjkl`，需要扭曲左手才能按到，非常费劲。用空格替代 `<c-w>` 后就舒服多了。

```
" leader键改为空格
nnoremap <space> <nop>
let mapleader = " "

" 关闭高亮
nnoremap <leader>n :nohlsearch<cr>
" 搜索文件
nnoremap <leader>f :Files<cr>
nnoremap <leader>b :Buffers<cr>
" 设置在分屏间移动的快捷键
nnoremap <leader>h <c-w>h
nnoremap <leader>l <c-w>l
nnoremap <leader>j <c-w>j
nnoremap <leader>k <c-w>k
" 设置移动分屏的快捷键
nnoremap <leader>H <c-w>H
nnoremap <leader>L <c-w>L
nnoremap <leader>J <c-w>J
nnoremap <leader>K <c-w>K
" 设置移动buffer的快捷键
nnoremap <leader>, :bprevious<cr>
nnoremap <leader>. :bnext<cr>
```

其中映射新按键的语句 `nnoremap` 仅作用于 normal 模式，且不会发生递归映射。关于各种 `map` 的介绍请见 [[Vim]vim的几种模式和按键映射](http://haoxiang.org/2011/09/vim-modes-and-mappin/)。

## 结语

我目前用到的配置就以上这些。可以说是很简陋了，自动补全、一键运行代码什么的统统没有。不过我觉得作为基本的文本编辑器已经够用了，如果读者有心得也可以传授我一下。

## 参考链接

[Vim 配置入门](https://www.ruanyifeng.com/blog/2018/09/vimrc.html)

[有哪些编程必备的 Vim 配置？](https://www.zhihu.com/question/19989337)

[上古神器Vim：从恶言相向到爱不释手 - 终极Vim教程01](https://www.bilibili.com/video/BV164411P7tw)

[iggredible/Learn-Vim](https://github.com/iggredible/Learn-Vim)
