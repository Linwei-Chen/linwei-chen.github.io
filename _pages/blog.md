---
permalink: /blog/
title: "Blog"
excerpt: ""
author_profile: true
---

# Blog

<p class="blog-featured-link">
  <strong>我的知乎专栏</strong>
  <a href="https://www.zhihu.com/column/c_1027873866512740352" rel="noopener">在知乎查看全部文章</a>
</p>

{% if site.posts.size > 0 %}
{% for post in site.posts %}
<div class="list__item">
  <article class="archive__item">
    <h2 class="archive__item-title">
      <a href="{{ post.url | relative_url }}" target="_self">{{ post.title }}</a>
    </h2>
    <p class="page__meta">
      <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y-%m-%d" }}</time>{% if post.tags and post.tags.size > 0 %} · {{ post.tags | join: " / " }}{% endif %}
    </p>
    <p class="archive__item-excerpt">{{ post.excerpt | strip_html | strip_newlines | truncate: 220 }}</p>
  </article>
</div>
{% endfor %}
{% else %}
暂无文章。新文章会发布在这里。
{% endif %}
