---
title: "解决博客 jsDelivr 资源无法访问的问题"
date: 2022-05-28
showToc: false
tags:
- hugo
- net
---

前段时间重看自己的文章时发现公式渲染、图片的放大缩小和代码高亮等功能都失效了，按 F12 发现原因是引自 `cdn.jsdelivr.net` 的字体资源、CSS 和 JS 文件都无法访问，挂梯子后页面恢复正常。

![website](/jsdelivr_problem/website.png)

<!--more-->

jsDelivr 是一款开源的免费公共 CDN，可以加速对 npm、GitHub 和 WordPress 上面静态资源的访问。通过 jsDelivr 引用网站所需的 CSS 和 JS 文件，可以避免直接向服务器请求资源，利用 CDN 加速网站的访问。然而，可能是 jsDelivr 提供的加速功能被一些用户拿来分发不和谐的内容等原因，2021 年 12 月 20 日，jsDelivr 在大陆的 CDN 节点被关闭，ICP 备案被注销，2022 年 4 月 28 日遭到 DNS 污染，自此大陆无法正常访问 jsDelivr，导致大批网站工作失常。jsDelivr 进出大陆的始末详见 [【杂谈】jsDelivr域名遭到DNS污染](https://luotianyi.vc/6295.html)。

据说很多人的博客因为缺失 CSS 文件而排版错乱，我使用的 Fuji 主题倒没有出现那么严重的错误，但公式失效还是令人非常恼火，这里就来解决一下这个问题。

我搜到的解决方法有三种：

- 使用 `cdn.jsdelivr.net` 未受污染的子域：
  - `fastly.jsdelivr.net`，由 Fastly 提供
  - `gcore.jsdelivr.net`，由 G-Core 提供
  - `testingcf.jsdelivr.net`，由 CloudFlare 提供
- 使用国内的静态库：
  - `cdn.staticfile.org`，七牛云和掘金的静态资源库
  - `cdn.bytedance.com`，字节跳动静态资源公共库
  - `cdn.baomitu.com`，360 前端静态资源库
- 将需要的静态资源下载到本地

第一种只需将博客主题的 HTML 文件中 jsDelivr 链接里的 `cdn` 替换为子域名即可；第二种需要在这些国内网站上搜索 JS 库的名字，然后复制搜索结果给出的链接，再替换掉对应的 jsDelivr 链接；第三种是替换为本地路径。为了方便和稳定，我使用的是国内的 `cdn.staticfile.org`。

在 VSCode 中搜索站点 `themes` 目录下含 `cdn` 的链接，收集得到

```
# KaTex 相关
https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.css
https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/katex.min.js
https://cdn.jsdelivr.net/npm/katex@0.13.11/dist/contrib/auto-render.min.js
# 搜索相关
https://cdn.jsdelivr.net/npm/art-template@4.13.2/lib/template-web.min.js
https://cdn.jsdelivr.net/npm/fuse.js@6.4.6/dist/fuse.min.js
# 页面相关
https://cdn.jsdelivr.net/npm/medium-zoom@1.0.6/dist/medium-zoom.min.js
https://cdn.jsdelivr.net/npm/lazysizes@5.3.2/lazysizes.min.js
https://cdn.jsdelivr.net/npm/prismjs@1.23.0/components/prism-core.min.js
https://cdn.jsdelivr.net/npm/prismjs@1.23.0/plugins/autoloader/prism-autoloader.min.js
```

省略了 APlayer、Google Analytics、Disqus 和字体的链接，前三者我用不到，而字体在 `staticfile.org` 上没搜到，就用备用字体算了。将上述链接修改为

```
# KaTex 相关
https://cdn.staticfile.org/KaTeX/0.15.6/katex.min.css
https://cdn.staticfile.org/KaTeX/0.15.6/katex.min.js
https://cdn.staticfile.org/KaTeX/0.15.6/contrib/auto-render.min.js
# 搜索相关
https://cdn.staticfile.org/art-template/4.13.2/lib/template-web.min.js
https://cdn.staticfile.org/fuse.js/6.6.2/fuse.min.js
# 页面相关
https://cdn.staticfile.org/medium-zoom/1.0.6/medium-zoom.min.js
https://cdn.staticfile.org/lazysizes/5.3.2/lazysizes.min.js
https://cdn.staticfile.org/prism/1.28.0/components/prism-core.min.js
https://cdn.staticfile.org/prism/1.28.0/plugins/autoloader/prism-autoloader.min.js
```

注意 KaTex 还要去 [官网](https://katex.org/docs/autorender.html) 把对应版本文件的哈希值复制过来，替换原来的 `integrity` 属性。至此博客又能在不挂梯子的情况下正常显示✌。如果有更好的方法（例如字体方面的）还请读者指点。

## 参考资料

[jsDelivr Wikipedia](https://en.wikipedia.org/wiki/JSDelivr)

[jsDelivr大面积失效，个人站点该怎么办？](https://blog.orangii.cn/2022/jsdelivr-alt/)