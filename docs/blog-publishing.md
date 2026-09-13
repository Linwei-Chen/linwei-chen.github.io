# 博客发布说明

博客文章使用 Markdown 编写，发布时不需要修改网页代码。

## 发布一篇文章

1. 复制 `_drafts/post-template.md` 到 `_posts/`。
2. 将文件名改为 `年-月-日-英文短标题.md`，例如 `2026-09-13-my-first-post.md`。
3. 修改文件顶部的标题、日期、标签和摘要，然后在下方编写正文。
4. 提交并推送到 GitHub；GitHub Pages 构建完成后，文章会自动出现在 `/blog/`。

## 更新已发布文章

直接编辑 `_posts/` 中对应的 Markdown 文件，提交并推送即可。文件顶部的 `title`、`date`、`tags` 和 `excerpt` 分别控制文章标题、日期、标签和博客列表摘要。

如果不想使用命令行，也可以在 GitHub 网页中打开文章文件，点击编辑按钮修改，最后提交更改。GitHub Pages 会自动重新构建网站。

文章示例：

```markdown
---
title: "我的第一篇文章"
date: 2026-09-13
tags:
  - 随笔
excerpt: "这是一段显示在博客列表中的摘要。"
---

这里是正文。
```

## 插入图片

将图片放在 `images/blog/` 中，然后在文章里引用：

```markdown
![图片说明](/images/blog/example.png)
```

## 保存草稿

尚未准备发布的文章可以保存在 `_drafts/` 中。该目录里的文章不会被正式网站构建和发布。
