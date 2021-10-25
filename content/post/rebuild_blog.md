---
title: "用 Hugo 重新搭建博客"
date: 2021-07-03
math: true
showToc: true
tags:
- hugo
- github
---

本博客之前是用软件 [Gridea](https://github.com/getgridea/gridea) 制作的，这是个静态博客写作客户端，可以作为 Markdown 编辑器，同时简单设置下就能一键生成静态页面并上传到网上，非常适合我这种电脑小白使用。不过前段时间发现怎么都没法上传本地写好的内容，于是决定重新用现在流行的 Hugo 来搭建博客。本文使用的是 0.84.4 版本的 Hugo 和 2.32.0 版本的 Git。

<!--more-->

## Hugo 的安装

Hugo 是一个由 Go 语言实现的静态网站生成器，因为听说使用起来比较简单，并且主题也多，所以选了它。二进制安装包可以直接在其 Github Releases 页面中下载到，我选择的是 `hugo_extended_0.84.4_Windows-64bit.zip`。新建一个目录 `bin`，将安装包里解压出来的东西都丢进去，然后把 `bin` 目录的路径添加到环境变量中，安装就完事了。以后直接在命令行中调用命令即可。

## Hugo 的基本用法

### 新建网站

在当前目录下新建网站

```bash
hugo new site ./ZhaJiMan.github.io
```

这样当前目录下会生成一个名为 `ZhaJiMan.github.io` 的网站目录，其结构为

```
.
├── archetypes      # 存放文章模板
├── config.toml     # 简单的配置文件
├── content         # 存放文章
├── data            # 存放生成静态页面时的配置文件
├── layouts         # 存放页面布局的模板
├── static          # 存放图片等静态内容
└── themes          # 存放下载的主题
```

之后的所有操作需要 `cd` 到这个目录下进行。

### 添加主题

主题可以在 [Hugo Themes](https://themes.gohugo.io/) 网站上找到，我选择的是自带 TOC 和评论功能的 [Fuji](https://github.com/dsrkafuu/hugo-theme-fuji)，通过 Git 命令安装。

```bash
git init
git submodule add https://github.com/WingLim/hugo-tania themes/hugo-tania
```

然后主题就会下载到 `themes` 目录中。一般主题的目录里都会含有一个 `exampleSite` 目录，顾名思义这是作者做好的示例网站，直接把里面的内容复制到网站根目录下，就能完成该主题最基本的配置，并实现示例网站的效果。之后修改根目录下的 `config.toml` 文件来自定义配置。

### 创建文章

Hugo 中的文章都以 Markdown 格式写作。在 `content/post` 目录下新建一个 Markdown 文件

```bash
hugo new post/rebuild_blog.md
```

默认的文章模板会使 Markdown 文件带有这样的开头

```yaml
---
title: "rebuild_blog"
date: 2021-07-03T16:47:34+08:00
draft: true
---
```

`---` 之间的内容服从 YAML 或 TOML 格式。`title` 即文章标题，默认与文件名相同；`date` 即日期时间；`draft` 表示该文章是否为草稿，如果是，那么后面生成静态页面时将不会含有该文章。此外还存在别的参数可供设置。`---` 之后的内容自然就是文章正文了。

Fuji 主题还额外强调要在正文中插入简介分割线 `<!--more-->`，以让文章列表的文章预览部分样式正确。

### 预览网站

建立一个本地服务器

```bash
hugo server
```

然后命令行会卡住，在浏览器内输入 [http://localhost:1313/](http://localhost:1313/) 预览网站，命令行内 Ctrl+C 关闭服务器。Hugo 的一个特色是可以进行动态预览，当你修改本地内容时，变化会马上反映在浏览器中的页面上。

### 生成静态页面

直接在生成在默认的 `public` 目录下

```bash
hugo
```

用 `-d` 参数可以指定目录，或者在配置文件里用 `publishDir` 参数指定默认的目录。

### 发布到 Github 上

这里用 Github Pages 来部署博客。首先在 `config.yaml` 里指定

```bash
publishDir: docs
```

然后再一个 `hugo` 命令，这样就把静态页面输出到 `docs` 目录下了。

接着在 Github 上以 `ZhaJiMan.github.io` 的名字（根据自己的用户名而定）新建一个空仓库，进行下面的 Git 命令

```bash
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/ZhaJiMan/ZhaJiMan.github.io.git
git push -u origin main
```

这段改编自空仓库页面出现的提示，大意是

- 将网站目录下的所有内容暂存。
- 把暂存的内容提交给版本库。
- 把主分支的名字从 `master` 改为 `main`。
- 添加远程仓库。
- 把本地内容推送到远程仓库里。

推送成功后，进入仓库的设置页面，点击侧栏的 Pages，再把 Source 选项改为 main 分支下的 docs 目录，这样 Github Pages 就会根据我们推送上去的 docs 目录里的静态页面来显示网站。这里指定 docs 的好处是还可以把网站的所有文件都备份到仓库里（不包含以 submodule 形式添加主题，详见参考链接）。最后在与仓库同名的网站 [https://zhajiman.github.io/](https://zhajiman.github.io/) 上看看自己的博客吧！

### 工作流

总结一下上面的流程

- 用 Markdown 写作。
- 用 `hugo server` 本地预览。
- 用 `hugo` 生成静态页面。
- 用 Git 的 `add`、`commit` 和 `push` 命令推送到网上。

## 其它功能

### 插入图片

以名为 `capslock.jpg` 的图片为例，将该图片放入 `static` 目录下，再在 Markdown 文件中以 `/capslock.jpg` 的路径引用即可。路径之所以写成这个形式，是因为 Hugo 会自动在图片路径前追加 `static` 的路径。为了区分开不同文章的用图，还可以在 `static` 下新建子目录，例如下面的写法

```markdown
![capslock](/rebuild_blog/capslock.jpg)
```

![capslock](/rebuild_blog/capslock.jpg)

其实这种隐式的路径在上一节中也频繁出现过。虽然 Hugo 可以解析这种路径，但 Markdown 编辑器不能，所以在编辑器的预览中会看不到图片。

### 渲染公式

Fuji 主题支持用 KaTex 渲染公式，使用方法为在文章开头或配置文件中添加 `math: true` 或 `katex: true`。使用过程中发现，KaTex 不能正常渲染行内公式，参考 KaTex 官网 [Auto-render Extension](https://katex.org/docs/autorender.html) 的例子，将 `themes/fuji/layouts/partials/math.html` 中的 KaTex 调用换成

```html
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css" integrity="sha384-Um5gpz1odJg5Z4HAmzPtgZKdTBHZdw8S29IecapCSB31ligYPhHQZMIlWLYQGVoc" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js" integrity="sha384-YNHdsYkH6gMx9y3mRkmcJ2mFUjTd0qNQQvY9VYZgQd7DcN7env35GzlmFaZ23JGp" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/contrib/auto-render.min.js" integrity="sha384-vZTG03m+2yp6N6BNi5iM4rW4oIwk5DfcNdFfxkk9ZWpDriOkXX8voJBFrAO7MpVl" crossorigin="anonymous"></script>
<script>
    document.addEventListener("DOMContentLoaded", function() {
        renderMathInElement(document.body, {
          delimiters: [
              {left: '$$', right: '$$', display: true},
              {left: '$', right: '$', display: false},
              {left: '\\(', right: '\\)', display: false},
              {left: '\\[', right: '\\]', display: true}
          ],
          throwOnError : false
        });
    });
</script>
```

这样行间公式与行内公式就都可以正常渲染。原理似乎是在函数 `renderMathInElement` 中指定识别公式的分隔符，不过具体细节我也不懂。本文便采用 KaTex 进行渲染，例如行内公式为 $e^{ix} = \cos{x} + i\sin{x}$，行间公式为
$$
P_e(\omega) = \frac{\hbar \omega^3}{4\pi^2 c^2} \frac{1}{\exp{(\hbar \omega / k_B T)} - 1}
$$

### 评论系统

Fuji 主题支持 Disqus、utterances 和 DisqusJS 三种评论系统，并且设置起来非常简单。这里采用依托于 Github issues 的 utterances。进入 [https://utteranc.es/](https://utteranc.es/)，按指示把 utterances app 安装到存储博客的仓库，然后在 `config.toml` 中设置

```toml
  utterancesRepo = "ZhaJiMan/ZhaJiMan.github.io"    # 格式为username/username/github.io
  utterancesIssueTerm = "pathname"                  # 以页面的pathname来生成issues
```

文章最下面就会出现评论区了，用 Github 账号登录即可发送评论。

### 设置网站图标

依据 Fuji 主页的说明，把自己喜欢的图片上传到 [https://realfavicongenerator.net/](https://realfavicongenerator.net/) 上，再把打包好的图标压缩包下载下来，解压到 `static` 目录中，接着把该网站提供的 HTML 代码粘贴到 `layouts/partials/favicon.html` 文件中，并修改一下 `href` 属性指向的路径即可。

### 显示文章点击量

这里使用 [不蒜子](http://ibruce.info/2015/04/04/busuanzi/) 实现统计。按官网和网上的介绍，首先需要在主题的 `head.html` 文件里加入脚本

```html
<script async src="//busuanzi.ibruce.info/busuanzi/2.3/busuanzi.pure.mini.js"></script>
```

再在主题的 `single.html` 文件中加入标签

```html
<span id="busuanzi_container_page_pv">
  本文总阅读量<span id="busuanzi_value_page_pv"></span>次
</span>
```

不过官网也提供另一种极简的标签

```html
本文总阅读量<span id="busuanzi_value_page_pv"></span>次
```

具体到我使用的 Fuji 主题上，先在 `themes/fuji/layouts/partials/head.html` 文件中加入脚本，再修改 `themes/fuji/layouts/_default/single.html` 文件，将标签加到文件第八行，post-meta（文章元数据）的块中

```html
<div class="post-item post-meta">
    {{ partial "post-meta.html" . }}
    <!-- 显示文章点击量 -->
    <span><i class="iconfont icon-time-sharp"></i>&nbsp;<span id="busuanzi_value_page_pv"></span>&nbsp;views</span>
</div>
```

其中 `<i class="iconfont icon-time-sharp">` 的部分是我从 `themes/fuji/layouts/partials/post-meta.html` 文件中抄来的，效果似乎是确定元数据的图标和字体。而标签则是用的前面提到的极简版。我试过直接将标签加到 `post-meta.html` 文件中，但会引起首页计数错乱；同时在 `footer.html` 中加入站点总点击量的尝试也失败了。我不懂 HTML，还请读者指导。

### 修改样式

依据 Fuji 主页的说明，利用 `assets/scss/_custom_var.scss` 文件修改 SCSS 变量（例如换颜色、换字体），利用 `assets/scss/_custom_rule.scss` 文件改写 SCSS 规则。

## 别人的博客

最后放两个别人用 Hugo + Fuji 搭的博客

[https://marcoscheel.de/post/2020/10/20201011-my-blog-has-moved/](https://marcoscheel.de/post/2020/10/20201011-my-blog-has-moved/)

[https://masatakashiwagi.github.io/portfolio/post/hugo-portfolio/](https://masatakashiwagi.github.io/portfolio/post/hugo-portfolio/)

## 参考链接

[如何使用Hugo在GitHub Pages上搭建免费个人网站](https://zhuanlan.zhihu.com/p/37752930)

[生物信息基础：实用Git命令，掌握这些就够了](https://zhuanlan.zhihu.com/p/315422417)

[hugo 导入图片，两种方式](https://blog.csdn.net/qq_38340601/article/details/108900666)

[single or double dollar sign as formula delimiter](https://github.com/KaTeX/KaTeX/issues/712)

[Git中submodule的使用](https://zhuanlan.zhihu.com/p/87053283)

[hugo建站 | 我的第一个博客网站](https://www.cnblogs.com/billie52707/p/13486133.html)