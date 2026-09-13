---
title: "从 L2P 到 Hash：视觉 Prompt 持续学习是怎样一步步修补自己的？"
date: 2026-09-13
tags:
  - 持续学习
  - 视觉 Prompt
  - 文献综述
excerpt: "梳理视觉 Prompt 持续学习从检索式 Prompt 池走向生成、保护与固定容量共享的完整研究脉络。"
---

> 一条从“像 RAG 一样检索 Prompt”，走向连续生成、训练—推理一致性、单 Prompt、零空间投影与共享 MoE 的研究路线。

视觉 Prompt 持续学习最吸引人的地方，可以用一句话概括：

**既然大型预训练 ViT 已经学到了足够丰富的视觉表示，持续学习时就不要反复改动整个模型；只训练一小段 Prompt，引导冻结模型调用已有能力。**

L2P 在 2022 年把这个想法真正带进了无回放持续学习：准备一个 Prompt 池，为每个输入检索若干 Prompt，再让冻结 ViT 完成分类。它看起来确实很像 RAG：Key 负责寻址，Prompt 负责提供被召回的“小块参数记忆”。[1]

但过去四年的发展也说明，L2P 只回答了第一个问题：**怎样找到一段 Prompt。** 它没有彻底回答另外五个问题：

1. 查表式 Top-K 是否真的能和分类目标一起学好？
2. 没有任务编号时，模型究竟错在选错 Prompt，还是错在全局分类头？
3. 为什么要为查询额外跑一次完整 ViT？
4. 每来一个任务就增加 Prompt，数千任务以后怎么办？
5. 如果只保留一个共享 Prompt，连续更新又怎样不覆盖旧知识？

这篇文章沿着这些问题，把 VPT、L2P、DualPrompt、S-Prompts、CODA-Prompt、DAP、APG、LAE、HiDe-Prompt、EvoPrompt、CPrompt、AdaPromptCL、OVOR、PGP、OS-Prompt、NoRGa、VPT-NSP²、CAPrompt、CPG、Additive Prompt Tuning、MQMK、VPT-NSP²++ 和 Hash 串成一张完整地图。

这里所说的“这一条线”，特指：**以冻结或主要冻结的预训练视觉 Transformer 为骨干，在无回放类增量或域增量学习中，通过视觉 Prompt 或与 Prompt 等价的轻量模块持续适配。** 检测、分割、视频、VQA、图学习等应用扩展不逐一展开；否则“标题里含 Prompt 的论文”会掩盖真正的机制演化。

![视觉Prompt持续学习的发展脉络](/images/blog/visual-prompt-continual-learning-lineage/lineage-v2.png)

这不是一条“后篇把前篇整体替掉”的单线，而是六条修补路线不断汇合：

| L2P 留下的问题 | 后续接力方法 | 核心变化 |
|---|---|---|
| 一个池混合共享与专属知识 | DualPrompt → AdaPromptCL | 先拆成共享/任务专属，再按真实语义组自适应扩容 |
| 硬 Top-K 太粗 | CODA-Prompt → MQMK | 从连续组件组合走向多任务 Query、类别级 Key |
| 固定模板不够细 | DAP/APG → EvoPrompt/CPG | 从“查 Prompt”走向“根据图片生成 Prompt” |
| 不知道到底错在哪里 | HiDe-Prompt → CPrompt/NoRGa | 拆开任务识别、任务内分类、全局校准，再修训练错配与门控 |
| 查询要多跑一次 ViT | OS-Prompt → Additive Prompt Tuning | 先改用中间 CLS 单次检索，再取消检索与 Token 拼接 |
| 每任务扩容无法走向数千任务 | OVOR/LAE/PGP/NSP² → CPG/Hash | 单 Prompt、快慢记忆、投影保护、固定专家池与历史感知共享 |

所以读下文时，不必记住所有缩写；只要抓住每篇论文接过的是哪一根“接力棒”。

## 一、先用同一个六分类例子读完全文

为了不让二十多篇论文变成名词接龙，全文只使用一个例子：

- 任务 1：学习“猫、狗”；
- 任务 2：学习“麻雀、鹰”；
- 任务 3：学习“汽车、卡车”；
- 学完任务 3 后，测试只给一张图片，**不告诉模型它来自哪个任务**；模型必须直接在六个类别里选答案。

这就是无回放类增量学习（Class-Incremental Learning，CIL）的核心困难。学习任务 3 时，旧的猫狗和鸟类图片已经不能再看；但测试时它们仍会出现。若模型先猜任务再分类，那么“把猫误判成任务 2”与“知道是任务 1、却把猫分成狗”是两种完全不同的错误。

域增量学习（Domain-Incremental Learning，DIL）则不同：类别可以仍是猫狗，只是图片依次来自照片、素描和红外相机。S-Prompts 主要解决这种“域发生变化”的问题，因此不能把它的数字和随机类别切分的 CIL 直接横比。[4]

### 1.1 先认清论文图里的五种零件

后面的图虽然画法不同，本质上反复出现五种东西：

1. **Prompt**：可训练的连续向量，不是中文或英文句子。它被插入 ViT，改变冻结主干处理图片的方式。
2. **Query**：从当前图片提取的“检索请求”。例如一张猫图经过冻结 ViT 得到的 CLS 特征。
3. **Key**：Prompt 的地址。Query 与哪个 Key 最相似，就倾向于调用哪个 Prompt。Key 通常也是可学习参数。
4. **Router**：比简单最近邻更一般的打分器。它可以同时给多个 Prompt 专家打分，再选 Top-K 或加权混合。
5. **Classifier**：最后把特征变成类别 Logit 的分类头。即使 Prompt 选对了，跨任务分类头仍可能偏向新类别。

一个最朴素的检索式流程是：

$$
\text{图片}\rightarrow\text{Query}\rightarrow\text{匹配Key}\rightarrow
\text{取Prompt}\rightarrow\text{带Prompt的ViT}\rightarrow\text{全类别分类头}.
$$

读每张方法图时都问六个问题：输入先到哪里？Prompt 从哪里来？是否要猜任务？Prompt 插在哪里？训练哪些参数？任务继续增加时保存什么？只要这六问能回答，论文就不再神秘。

### 1.2 “参数少”不等于“能无限扩展”

无回放只表示不保存旧图片，不表示不保存任何历史。旧 Prompt、Key、Router、分类头、类别原型、均值和协方差都可能继续累积。因此要核算：

$$
\text{总资源}(T,C)=P_{\text{Prompt}}(T)+P_{\text{Router}}(T)+P_{\text{Head}}(C)+M_{\text{stats}}(C)+\text{FLOPs},
$$

其中 $T$ 是任务数，$C$ 是累计类别数。一篇论文即使说“Prompt 只占主干的 0.1%”，也可能仍然每来一个任务就增加 Router，或为了找 Prompt 多跑一遍 ViT。

## 二、技术原点 VPT：给冻结 ViT 插入可学习 Token

**一句话理解：** VPT 不重新训练整台 ViT，而是在它的输入序列中插入少量可学习向量；后续持续学习方法争论的，主要是“这些向量从哪里来、应该选哪一组、怎样不遗忘”。VPT 本身并不是持续学习算法，而是这条路线的结构接口。[0]

### 2.1 先看论文方法图

先看图例：雪花表示冻结，火焰表示可训练。再比较中间的 VPT-Deep 和右边的 VPT-Shallow，关键不是图片怎么切块，而是彩色 Prompt Token 在多少层出现。

![VPT论文方法图：Deep与Shallow Prompt](/images/blog/visual-prompt-continual-learning-lineage/00-vpt-fig2.png)

*图源：VPT 原论文 Figure 2（PDF 第 4 页）。*

跟图走一遍：

1. 左侧图片被切成 Patch，每个 Patch 变成一个视觉 Token，再加一个用于汇总全图的 CLS Token。
2. **VPT-Shallow** 只在第一层输入处加入 Prompt。它们影响第一层后便不再重新注入。
3. **VPT-Deep** 在多个 Transformer 层各自加入一组 Prompt，因此每一层都能获得专门的可学习引导。
4. ViT 主干保持冻结；只有 Prompt 和最末端分类头更新。这样训练参数很少，又能借用预训练 ViT 的能力。

一张图被切成 Patch 后，普通 ViT 输入为：

$$
Z_0=[x_{\mathrm{CLS}};x_1;x_2;\ldots;x_N]+E_{\mathrm{pos}}.
$$

VPT 加入 $L_p$ 个可学习 Token：

$$
Z'_0=[p_1;p_2;\ldots;p_{L_p};x_{\mathrm{CLS}};x_1;\ldots;x_N]+E_{\mathrm{pos}}.
$$

放回六分类例子：如果只做普通 VPT，我们可以在猫狗数据上学一组 Prompt；可到了鸟类任务，继续改同一组 Prompt 可能覆盖猫狗知识，另建一组又需要在测试时先判断该用哪组。**这两个问题正是 Prompt 持续学习的起点。**

### 2.2 Prompt Token 与 Prefix Prompt 不是完全相同的接法

普通 Prompt Token 与 CLS、Patch 一起进入注意力，会参与 Query、Key、Value 的计算。许多后续论文采用 Prefix 形式，把 Prompt 拆成 $P_K,P_V$，只接到注意力的 Key 和 Value：

$$
\operatorname{Attn}\bigl(Q,[P_K;K],[P_V;V]\bigr).
$$

这个差别在 NoRGa 中尤其重要：Prefix 的每个 Value 都像一个候选输出，注意力分数则像专家门控。NoRGa 因而把 Prefix Prompt 重新解释为注意力内部的隐式 MoE，而不是把 ViT 的 FFN 换成 MoE。

## 三、L2P：把持续学习改造成“像 RAG 一样检索 Prompt”

**一句话理解：** 先用图片产生 Query，在固定 Prompt 池中查找最匹配的 Key，只把命中的几段 Prompt 插进 ViT。[1]

**它要修复的问题：** 普通 VPT 只有一组 Prompt。连续学习鸟类时若直接更新它，可能覆盖猫狗知识；如果人为给每个任务一组 Prompt，测试时又不知道该拿哪一组。L2P 的答案是：不要依赖任务编号，让图片内容自己检索。

### 3.1 从论文图完整走一遍

这张图分左右两半。左半回答“怎样从池中取 Prompt”，右半回答“取出后怎样分类”。注意，彩色 Key 和下方同色 Prompt 是一一配对的；灰色项没有被选中。

![L2P论文方法图：Key-Prompt池、查询与Top-K检索](/images/blog/visual-prompt-continual-learning-lineage/01-l2p-fig2.png)

*图源：L2P 原论文 Figure 2（PDF 第 4 页）。*

按照箭头顺序：

1. 模型预先准备 $M$ 个 Prompt，每个 Prompt 配一个可学习 Key：

$$
   \mathcal P=\{P_1,\ldots,P_M\},\qquad
   \mathcal K=\{k_1,\ldots,k_M\}.
$$

2. 当前图片先经过一个冻结的查询函数，得到全图特征 $q(x)$。标准实现使用无 Prompt ViT 的 CLS。
3. 将 $q(x)$ 与所有 Key 做余弦相似度，选 Top-K：

$$
   I(x)=\operatorname{TopK}_j\cos(q(x),k_j).
$$

4. 根据 Key 的编号取出配对 Prompt。图里只取彩色命中项，**不会让池中所有 Prompt 都处理整张图**。
5. 把这些 Prompt Token 拼在 CLS 和 Patch Token 前，再运行带 Prompt 的冻结 ViT。
6. 最后的全类别分类头在所有已见类别中给答案。学到任务 3 时，它输出猫、狗、麻雀、鹰、汽车、卡车共六个 Logit，而不是只输出当前两类。

放进例子：一张猫图生成的 Query 可能最接近 Key 2 和 Key 7，于是取出 $P_2,P_7$；一张卡车图则可能取 $P_4,P_9$。Prompt 槽位与任务不是强绑定的，所以猫和麻雀也可能共享一部分 Prompt。

### 3.2 Key 和 Prompt 分别怎样学习？

Key **确实是可学习向量**。官方 JAX 代码用 `self.param("key", ...)` 注册它，再执行归一化、矩阵乘法和 `jax.lax.top_k`。分类损失训练被选中的 Prompt 与分类头；匹配损失把命中的 Key 拉近当前 Query：

$$
\mathcal L=\mathcal L_{\mathrm{CE}}+
\lambda\sum_{j\in I(x)}\bigl(1-\cos(q(x),k_j)\bigr).
$$

可以把它类比为 RAG，但要加两个限定：传统 RAG 检索的是已有文档，L2P 检索的是可学习参数；文档通常有明确文本含义，而某个 Prompt 槽位未必对应一个人能命名的任务。

### 3.3 为什么标准 L2P 真的要算两遍 ViT？

这里存在先后依赖：必须先理解图片才能选 Prompt；但选出的 Prompt 又要从 ViT 前部参与图片计算。官方训练和评估代码都先调用无 Prompt 的 `original_vit_model` 取得 `pre_logits`，再将其作为 `cls_features` 传给带 Prompt 模型。所以标准 `embedding_key="cls"` 路径确实是两次模型调用。

![L2P两次前向与OS-Prompt一次前向的直观差别](/images/blog/visual-prompt-continual-learning-lineage/two-pass-vs-one-pass.png)

第一次：

$$
x\xrightarrow{\text{无Prompt ViT}}q(x).
$$

第二次：

$$
[P_{i_1};\ldots;P_{i_K};x]\xrightarrow{\text{带Prompt ViT}}\hat y.
$$

为什么不能第一次只算 CLS？因为 ViT 的 CLS 从第一层起就与全部 Patch Token 做自注意力；最终 CLS 不是一条能脱离 Patch 独立计算的旁路。后面的 OS-Prompt 会通过“走到中间层再检索”安全地消除第二次完整前向。

### 3.4 L2P 真正解决了什么，又留下什么？

它第一次把“测试时不知道任务”改写为实例级内容寻址，而且 Prompt 池大小预先固定，**不是每个任务新增一个 Prompt**。但它留下四个明确问题：

- 硬 Top-K 只更新命中槽位，Key 的匹配目标与最终分类目标也并不完全一致；
- 同一个池既要存跨任务共性，又要存任务差异，分工含混；
- CLS 查询导致两次 ViT；
- 固定池不会随任务数增长，却可能被越来越多知识挤满。

后面的 DualPrompt、CODA-Prompt、OS-Prompt 和固定容量路线，分别从这些缺口出发。

## 四、2022：DualPrompt 与 S-Prompts——先把共享和专属拆开

### 4.1 DualPrompt：G-Prompt 管共性，E-Prompt 管个性

**一句话理解：** L2P 把所有记忆混在一个池里；DualPrompt 把它拆成“大家共用的基础课”和“每个任务自己的专业课”。[2]

**它修复 L2P 的问题：** 同一批 Prompt 同时承担共享知识与任务特有知识，训练目标容易互相拉扯。DualPrompt 明确分工：

- G-Prompt：所有任务共享，负责稳定的通用指令；
- E-Prompt：每个任务一套，负责专属知识。

先看论文图。左半是完整架构，右半放大了 Prefix Prompt 怎样拼到注意力的 Key/Value。重点看蓝色 G-Prompt 与彩色 E-Prompt 被放在不同层。

![DualPrompt论文方法图：共享G-Prompt与任务E-Prompt](/images/blog/visual-prompt-continual-learning-lineage/02-dualprompt-fig2.png)

*图源：DualPrompt 原论文 Figure 2（PDF 第 6 页）。*

跟图走一遍：

1. 图片先产生 Query。训练时任务编号已知，可直接选当前任务的 E-Prompt；测试时不知道任务，才用 Query 与任务 Key 匹配。
2. G-Prompt 不需要检索，任何图片都会使用，负责跨任务通用信息。
3. E-Prompt 由任务 Key 选择，负责猫狗、鸟类或车辆的专属适配。
4. 两类 Prompt 可以插入不同 Transformer 层。论文右图展示的是 Prefix 接法：把 Prompt 只接到注意力 Key/Value，而非简单放在输入序列前。
5. 最后仍用全类别头输出六类结果。

若第 $t$ 个任务的专属 Prompt 为 $e_t$、任务 Key 为 $k_t$，训练时任务编号已知，直接使用 $e_t$；测试时再用冻结 ViT 的 CLS 查询所有任务 Key：

$$
\hat t=\arg\max_i\cos(q(x),k_i),qquad q(x)=f_{\theta}(x)[\mathrm{CLS}].
$$

例子里，任务 1 建立 $E_1$，任务 2 再建 $E_2$，任务 3 再建 $E_3$。一张猫图若错匹配到车辆 Key，就会把 $E_3$ 插进网络；这说明参数隔离减少了彼此覆盖，却把风险转移成任务路由错误。

它真正解决的是“共享与专属分工”，代价也非常直接：

$$
P_{\mathrm{E-Prompt}}(T)=O(T).
$$

任务越多，专属 Prompt 与 Key 越多；标准 CLS 检索仍需额外查询前向。DualPrompt 因此开启了一个影响深远、又不断被质疑的范式：**用参数隔离换更少遗忘。**

### 4.2 S-Prompts：在域增量里，把任务当作域专家

**一句话理解：** 先判断图片来自照片域、素描域还是红外域，再调用该域自己的 Prompt 专家。[4]

**它与 DualPrompt 的差别：** DualPrompt 面向类增量，任务常是不同类别组；S-Prompts 主要面向域增量，同一类别在不同风格或传感器下发生变化。域通常比“随机分的一组类别”更有稳定语义，所以每域一套 Prompt 更自然。

先看图：上方图像支路先走无 Prompt 编码器做域识别，选中 Prompt 后再编码；下方语言支路只属于结合 CLIP 的 S-liPrompts 扩展。

![S-Prompts论文方法图：域中心检索与图像语言Prompt](/images/blog/visual-prompt-continual-learning-lineage/03-sprompt-fig2.png)

*图源：S-Prompts 原论文 Figure 2（PDF 第 4 页）。*

跟图走一遍：

1. 学照片域时，训练照片 Prompt，并对无 Prompt 的冻结图像特征做 K-means，保存若干照片域中心。
2. 学素描域时，冻结照片 Prompt，新增素描 Prompt，并保存素描中心；后续域重复这一过程。
3. 测试图片先通过无 Prompt 编码器得到特征，用 KNN 找最近的域中心。
4. 最近中心属于哪个域，就取哪个图像 Prompt，再跑一次带 Prompt 编码器进行分类。
5. S-liPrompts 还取对应语言 Prompt 生成类别文本特征，让图像与文本共同决策。

例如同一只猫既可能是照片，也可能是素描。S-Prompts 先判断“这是素描风格”，再用素描 Prompt 判断“这是猫”。它解决的是域路由与域专属适配，不是随机六分类例子中的通用跨任务类别混淆。

其收益是旧域 Prompt 冻结后不会被新域直接覆盖；代价是 Prompt 和域中心都随域数增长，而且测试包含一次无 Prompt 域识别和一次带 Prompt 分类。它揭示了一个后来非常重要的问题：**只有当任务边界真的对应稳定语义域时，每任务隔离参数才有充分理由。** OVOR 正是从随机 CIL 中的相反情况提出质疑。

## 五、2023：三种根本反思——怎样组合、怎样生成、到底错在哪里

### 5.1 CODA-Prompt：从硬检索变成连续组合

**一句话理解：** L2P 像从工具箱硬取两把完整工具；CODA-Prompt 则把许多 Prompt 组件按连续权重调成一份“配方”。[3]

**它修复的问题：** L2P 的 Top-K 是离散选择，未选组件得不到当前样本的分类梯度；DualPrompt 又把任务与一整套 Prompt 硬绑定。CODA-Prompt 对此提出两个批评：

1. Key 与 Prompt 的选择没有被最终任务损失端到端地充分优化；
2. 每个任务只分到一个硬 Prompt，任务复杂度变大时，增加 Prompt 长度很快出现收益饱和。

先看图：左侧不是“一行一个完整 Prompt”，而是三张并排的组件表——Prompt 组件 $P$、Key $K$、注意力向量 $A$。中间的逐元素乘法和余弦相似度得到连续权重，右边矩阵乘法才合成最终 Prompt。锁表示旧组件冻结，新增彩色条表示当前任务扩容。

![CODA-Prompt论文方法图：组件级连续组合](/images/blog/visual-prompt-continual-learning-lineage/04-codaprompt-fig2.png)

*图源：CODA-Prompt 原论文 Figure 2（PDF 第 5 页）。*

跟图走一遍：

1. 图片先生成 Query $q(x)$。
2. 每个组件都有注意力向量 $A_m$，先用 $q(x)\odot A_m$ 屏蔽与该组件无关的特征维度。
3. 处理后的 Query 与组件 Key $K_m$ 做余弦匹配，得到连续系数 $\alpha_m$。
4. 所有组件按 $\alpha_m$ 加权相加，而不是只复制一个完整槽位。
5. 合成 Prompt 插入多个 Transformer 层，分类损失可以穿过加权过程训练新组件的 $P,K,A$。

公式是：

$$
\alpha_m(x)=\cos\bigl(q(x)\odot A_m,K_m\bigr),
$$

$$
P(x)=\sum_{m=1}^{M}\alpha_m(x)P_m.
$$

$A_m$ 让每个组件只关注查询的某些维度；连续权重让多个组件能够共同贡献，也让分类损失可以穿过组合过程更新 Key、注意力向量和 Prompt。

放回例子：猫图可以使用 60%“动物外形”组件、30%“毛发纹理”组件和 10%“背景”组件；麻雀图可能复用动物外形，却提高羽毛组件权重。它比“一张图只属于某个任务 Prompt”更柔软。

为了避免新组件覆盖旧知识，进入新任务时冻结旧组件，只扩展一组新组件，并对 $P,K,A$ 使用正交正则：

$$
\mathcal L_{\mathrm{ortho}}(B)=\|BB^\top-I\|_2.
$$

CODA 真正解决了“怎样连续组合、怎样让分类损失更直接地优化寻址”。但它没有解决双前向，也明确选择用扩容换可塑性：每个任务增加一组组件，长期容量仍是 $O(T)$。

### 5.2 DAP：不再取固定模板，而是为每张图生成 Prompt

**一句话理解：** 猫和狗即使属于同一任务，也不必共用完全相同的 Prompt；DAP 让生成器根据当前图片现做一份实例级 Prompt。[5]

**它修复 CODA/L2P 的问题：** 池中的 Prompt 无论怎样组合，基本零件仍是有限的固定模板。当新数据与预训练域差得很远，组级模板可能不够细。

论文图左半是外部流程，右半把生成器 $G$ 拆开。先看左半：CLS 与 Patch Token 都送进 $G$，产出一组只属于当前图片的 Prompt；右半则解释 $G$ 怎样把长 Token 序列压成短 Prompt。

![DAP论文方法图：实例级Prompt生成器](/images/blog/visual-prompt-continual-learning-lineage/12-dap-fig2.png)

*图源：DAP 原论文 Figure 2（PDF 第 2 页）。*

跟图走一遍：

1. 图片先变成 CLS 与 Patch Token，记为 $E\in\mathbb R^{(N+1)\times D}$。
2. 生成器先做 LayerNorm 和转置，让 MLP 沿 Token 维压缩：几百个图像 Token 被变成 $L_p$ 个 Prompt Token。
3. 任务条件嵌入再产生缩放 $\gamma_e$ 与偏移 $\beta_e$，对结果做仿射调节：

$$
   P^a=\Bigl(\gamma_e\,\operatorname{MLP}(\operatorname{LN}(E)^\top)+\beta_e\Bigr)^\top.
$$

4. 生成的 Prompt 在若干层中使用，引导冻结 ViT 输出全类别预测。

放回例子：一张侧脸猫与一张夜间猫图会得到不同 Prompt，而不只是共同取“任务 1 Prompt”。但 DAP 仍用任务 Key 决定条件嵌入，所以测试时依然要判断任务，任务 Key 仍随 $T$ 增长。它解决了**表达粒度**，没有完全解决**任务路由与长期状态增长**。

### 5.3 APG：走到 ViT 中间，再根据已经形成的语义生成 Prompt

**一句话理解：** DAP 在入口处根据原始 Token 生成 Prompt；APG 先让图片在冻结 ViT 中走一段，等中层 CLS 已经看懂一些内容后，再用交叉注意力生成后半程 Prompt。[6]

**它修复的问题：** 早期 Prompt 方法默认预训练查询已经足够可靠；当目标域与 ImageNet 差异大时，早期查询和静态 Prompt 都可能失效。

先看图的三个区域：上方是完整流水线；左下是 Prompt 生成器；右下是用来防遗忘的类别高斯知识池。三者分别回答“在哪里插”“怎样生成”“怎样记住旧类”。

![APG论文方法图：中层交叉注意力Prompt生成](/images/blog/visual-prompt-continual-learning-lineage/13-apg-fig2.png)

*图源：APG 原论文 Figure 2（PDF 第 4 页）。*

跟图走一遍：

1. 图片先无 Prompt 地通过 ViT 前半段，中层 CLS 变成查询 $z$。
2. 一个候选 Prompt 列表 $I_t$ 同时作为交叉注意力的 Key 和 Value；$z$ 决定每个候选贡献多少。
3. 多头交叉注意力合成实例 Prompt：

$$
   \widetilde P=\operatorname{MMHA}(z,I_t,I_t).
$$

4. 生成 Prompt 插入后半段 ViT，产生分类特征。
5. 学过某类后，APG 保存该类特征的均值和协方差。学习汽车/卡车时，它可从猫狗、鸟类的高斯分布采样伪特征，继续约束生成器和全类别分类头。

这相当于“先读半本书，再决定后半本需要哪张书签”。它只需一条分段前向，不需要先跑完整查询 ViT 再从头跑第二次；但候选 Prompt 列表可以扩展，类别统计为 $O(C)$，而高斯伪特征也不等价于真实旧图。

DAP 与 APG 共同完成一次观念转换：**Prompt 不必是内存里被检索的一行参数，也可以是条件函数根据输入即时生成的输出。**

### 5.4 HiDe-Prompt：别只盯着“选错 Prompt”，CIL 至少有三种错误

**一句话理解：** 总准确率下降不一定是 Prompt 没学好；还可能是任务猜错，或不同任务的分类分数不可直接比较。HiDe-Prompt 把这三种错误拆开诊断和修复。[8]

先看论文图，但不要把它误读成一条从左到右的普通网络。图中 TII、WTP、TAP 是三个问题层级；两个 Statistics 存储桶分别保存无 Prompt 与带 Prompt 的类别统计。

![HiDe-Prompt论文方法图：TII、WTP与TAP三层分解](/images/blog/visual-prompt-continual-learning-lineage/05-hideprompt-fig3.png)

*图源：HiDe-Prompt 原论文 Figure 3（PDF 第 6 页）。*

先用猫图说明三种错法：

- **WTP，任务内分类：** 已知这是任务 1，能否在猫和狗之间选对？这主要检验当前任务 Prompt 与局部分类能力。
- **TII，任务身份推断：** 不给任务编号，能否先判断它来自猫狗任务，而不是鸟类或车辆任务？
- **TAP，全任务自适应预测：** 即使任务判断和局部分类不错，把六类 Logit 放在一起时，是否因为新旧任务分数尺度不同而把猫判成汽车？

它把类增量预测写成：

$$
P(x\in X_{i,j})=
\underbrace{P(x\in X_{i,j}\mid x\in X_i)}_{\text{WTP：任务内分类}}
\underbrace{P(x\in X_i)}_{\text{TII：任务身份推断}}.
$$

跟图走完整训练流程：

1. 每个任务保存一组 Prompt。新任务 Prompt 从前一任务初始化，并混合旧 Prompt 促进迁移。
2. 用正确任务 Prompt 学好 WTP，让猫狗内部、鸟类内部、车辆内部先分清。
3. 对**无 Prompt 特征**按类别估计高斯分布，用采样伪特征持续训练 TII 头，学会判断输入来自哪个任务。
4. 对**带 Prompt 特征**再保存一套类别分布，用伪特征重训或校准全类别头，处理 TAP 的跨任务 Logit 偏置。

测试流程因此很清楚：

$$
x\xrightarrow{\text{无Prompt特征}}\hat t
\xrightarrow{P_{\hat t}}\text{带Prompt特征}
\xrightarrow{\text{全类别头}}\hat y.
$$

HiDe 真正解决的是**诊断框架与全局校准**，而不是发明一种更复杂的 Prompt Token。代价是每任务 Prompt 为 $O(T)$，两套类别统计为 $O(C)$，推理仍含无 Prompt 任务识别与带 Prompt 分类两阶段。后面的 CPrompt 改训练错配，NoRGa 改 Prompt 门控，但都直接受益于这套分解。

### 5.5 LAE：把 Prompt 提升为通用 PEFT 问题

**一句话理解：** 与其每个任务新建 Prompt，不如保留一个“学得快的在线模块”和一个“忘得慢的长期模块”；而且这套办法不只适用于 Prompt，也适用于 Adapter、Prefix 和 LoRA。[7]

**它修复的问题：** 前面的方法太关注 Prompt 的形状，却忽略了持续覆盖是所有参数高效微调（PEFT）模块的共同问题。

先看图：左侧是新任务怎样进入 Online 模块并积累到 Offline 模块；中间列出三种可替换载体；右侧是测试时两个模型共同投票。

![LAE论文方法图：在线与离线PEFT模块](/images/blog/visual-prompt-continual-learning-lineage/16-lae-fig2.png)

*图源：LAE 原论文 Figure 2（PDF 第 4 页）。*

跟图走一遍：

1. 冻结同一个预训练主干，只挂载少量可训练参数。它可以是 Adapter、Prefix Prompt 或 LoRA，不限定为某一种 Prompt。
2. Online 模块直接在当前任务上更新，快速学会汽车/卡车，优点是可塑性强，缺点是容易偏向最新任务。
3. Offline 模块不追逐每一步梯度，而是用在线权重的指数移动平均缓慢更新，积累猫狗、鸟类与车辆的长期知识。
4. 测试时同一图片分别经过 Online 与 Offline 两个轻量专家，再集成预测。

LAE 不是在解决 Key 检索，也不需要为每个任务永久保存一套模块。它把核心矛盾改写为“快系统怎样吸收新知识、慢系统怎样稳定保存”。代价是推理需要两个 PEFT 分支，并且指数平均无法保证互相冲突的任务都被无损保留。

## 六、2024：路线开始分叉，而不是继续堆更大的池

### 6.1 EvoPrompt：把 Prompt 池压进一个连续 FFN 记忆

**一句话理解：** 不再存一张离散的 Key-Prompt 表，而是用一个小 FFN 把图片特征连续映射成 Prompt；它像把查表记忆压缩成一段函数。[9]

**它修复的问题：** 硬检索有路由断点，逐任务扩充又会无限增长。EvoPrompt 希望用固定形状的函数记忆，同时为不同图片产生不同 Prompt。

论文图需要分两半读：左半解释“FFN 为什么等价于连续 Prompt 记忆”，右半解释“新任务训练后的工作记忆怎样与长期参考记忆对齐并融合”。

![EvoPrompt论文方法图：连续FFN记忆与对齐融合](/images/blog/visual-prompt-continual-learning-lineage/06-evoprompt-fig1.png)

*图源：EvoPrompt 原论文 Figure 1（PDF 第 3 页）。*

跟图走一遍：

1. 图片 Query 乘 $W_k$ 后经过 ReLU，得到一组输入相关的非负激活；它们类似对许多隐式 Key 的软选择。
2. 激活再乘 $W_v$，得到一条长向量，重排后成为多个 Prompt Token：

$$
P_b(x)=\operatorname{ReLU}\bigl(q(x)W_k^{(b)}\bigr)W_v^{(b)}.
$$

3. $W_k$ 像 Key，隐藏激活像连续门控，$W_v$ 像 Prompt Value；但整个过程可微，不需要先决定“这是任务 1 还是任务 3”。隐藏宽度 $d$ 就是记忆容量。
4. 当前任务先更新 Working Prompt Memory；Reference Prompt Memory 保存此前的长期版本。
5. 任务结束时先对齐两个 FFN 的隐藏节点，再按重要性融合到新的参考记忆。

为什么一定要“先对齐再平均”？两个 FFN 即使功能相同，隐藏节点的排列也可以不同。直接把第 1 个节点逐元素平均，可能是在混合两个完全不同的功能。论文用最优传输寻找功能相近节点的对应关系，再融合：

$$
W_g^{t+1}=\lambda\widehat W_f^t+(1-\lambda)W_g^t.
$$

在例子中，工作记忆先学车辆，参考记忆保存动物；对齐后，“轮廓”节点可以互相融合，而不会把“羽毛”节点误平均到“车轮”节点。最终 Prompt 记忆形状固定，不为每个任务永久保存一套 Prompt。它用**融合误差**替换**参数增长**：容量有界，但互相冲突的知识必须被压进同一函数。

### 6.2 CPrompt：训练时故意制造测试时会遇到的错误

**一句话理解：** 训练时不能永远给模型正确 Prompt 和正确分类头，否则测试时一旦路由错，模型会毫无抵抗力；CPrompt 在训练阶段主动制造这种错误。[10]

**它修复 HiDe 揭示的问题：** 许多方法训练时知道任务编号，总用正确 Prompt；测试时却要检索 Prompt，并在全部历史类别中比较 Logit。训练条件和测试条件不一致。

论文图左右是两种一致性训练。左边让同一特征面对所有历史分类头，右边故意给当前分类器随机历史 Prompt。

![CPrompt论文方法图：分类器一致性与Prompt一致性](/images/blog/visual-prompt-continual-learning-lineage/07-cprompt-fig3.png)

*图源：CPrompt 原论文 Figure 3（PDF 第 3 页）。*

先读左半：

**分类器不一致。** 学车辆时，如果只训练车辆头，它从未学过自己的 Logit 将来要与猫狗头、鸟类头竞争。CPrompt 将当前 Prompt 得到的特征送入全部历史分类器，并用平滑正则约束非当前头的响应，提前模拟全局比较。

再读右半：

**Prompt 不一致。** 训练时总给车辆图正确的 $P_3$，测试时 Key 却可能误选 $P_1$。于是训练当前分类器时，从所有已见 Prompt 中均匀随机采一个：

$$
P_i,\quad i\sim\operatorname{Uniform}(1,t),
$$

并要求当前分类器即使接到错误 Prompt 也能认出汽车/卡车。由于正确 Prompt 只有 $1/t$ 的概率被抽到，图中辅助分类器专门使用当前正确 Prompt，避免它反而学不充分。

此外，它不再用一个 Key 概括整项任务，而是每类一个 Key。查询先找最近类别 Key，再使用其所属任务的 Prompt。

CPrompt 的思想非常实用：**与其假设路由永不犯错，不如让下游在训练时见过路由错误。** 它改善的是容错与 Logit 可比性，不是减少存储；每任务 Prompt 与每类 Key 仍分别为 $O(T)$、$O(C)$。

### 6.3 AdaPromptCL：共享一个还是每任务一个？答案应由语义漂移决定

**一句话理解：** 不按任务编号机械增加 Prompt，而是把语义相近的任务归为一组，共享组 Prompt；只有真正出现新语义时才扩容。[11]

**它修复的问题：** One-Prompt 容量可能不够，One-Prompt-per-Task 又浪费且阻碍相似任务迁移。AdaPromptCL 观察到两种极端策略都有问题：

- 所有任务共享一个 Prompt：适合温和、相似的语义变化，但遇到突变可能容量不足；
- 每任务独立 Prompt：适合互不相关的任务，但相似任务之间无法正迁移，还浪费参数。

论文图不是普通神经网络图，而是一条随任务到来的时间线。从左到右看“新任务分配”“预留可能组合”“后来重新细化”“最终取回 Prompt”。

![AdaPromptCL论文方法图：语义组分配与后续细化](/images/blog/visual-prompt-continual-learning-lineage/08-adapromptcl-fig4.png)

*图源：AdaPromptCL 原论文 Figure 4（PDF 第 5 页）。*

跟图走一遍：

1. 新任务到达后，先用任务表示与已有语义组比较。
2. 若足够相似，就加入最近组并共享、更新该组 Prompt；若差异明显，就创建新组。
3. 只看早期任务可能分错，所以系统还保留“前瞻组”候选。例如猫狗与鸟类刚出现时都像动物，可能暂时合组；车辆出现后，任务间关系变得更清楚。
4. 后续证据触发组结构细化或重组；测试时先定位语义组，再取回组 Prompt。

在例子中，猫狗和鸟类可以共享“动物组 Prompt”，车辆另建一组，而不是固定保存三组。若任务 4 又是野生动物，它可复用动物组；若是医学影像，则创建新组。

其容量是 $O(G)$，$G$ 为最终语义组数，通常小于任务数，但若每个任务都独特，最坏仍是 $O(T)$。它把扩容规则从“来了一个任务”改成“发现一种新颖语义”，更合理，却依赖分组质量，也要维护和修正候选组。

### 6.4 OVOR：也许根本不需要猜任务，一个 Prompt 就够

**一句话理解：** 如果任务只是随机把类别切成几组，也许根本没有值得区分的“任务语义”；那就始终更新同一个 Prompt，把主要精力用在修正全局分类边界。[12]

**它反驳的前提：** DualPrompt、HiDe 等默认不同任务需要不同 Prompt。OVOR 指出，在随机类增量基准里，“任务 3”和“任务 7”未必对应真实域，为它们分别准备 Prompt 可能是在人工制造任务识别错误。

论文图分三行，按训练阶段从上到下读：第一行训练共享 Prompt；第二行冻结 Prompt 后估计类分布并生成虚拟离群点；第三行只用真实特征与离群点继续校准分类器。

![OVOR论文方法图：OnePrompt与虚拟离群点正则](/images/blog/visual-prompt-continual-learning-lineage/09-ovor-fig2.png)

*图源：OVOR 原论文 Figure 2（PDF 第 6 页）。*

跟图走一遍：

1. 整个任务序列只有一个 Prompt $P$。学猫狗后继续在鸟类上更新同一个 $P$，再继续学车辆：

$$
\mathcal P=\{P\},\qquad P(x)=P.
$$

2. 没有 Prompt Key、Top-K 或任务路由，图片只需一次带 Prompt 前向。
3. 当前任务训练完成后冻结 Prompt，在当前类特征周围拟合分布并采样候选点。
4. 只保留远离真实类中心的点作为 Virtual Outlier。它们不代表具体旧图，而是“这里不应高置信属于当前类”的边界外区域。
5. 第二阶段只更新分类器：真实汽车特征应高置信属于汽车/卡车，虚拟离群点则应保持低置信，减少新任务分类头占领旧类区域。

论文还发现相邻任务后的 Prompt 权重余弦相似度很高，在已知正确任务时旧任务性能下降不大，说明真正瓶颈可能是跨任务类别混淆，而非 Prompt 表示立刻崩坏。其 OnePrompt 在论文 10 任务 ImageNet-R 设置中超过 L2P、DualPrompt，并将推理主干开销约减半。

OVOR 的历史意义是提供固定容量反例：**多 Prompt 并非天然更强；先问任务边界是否有语义，再决定是否隔离参数。** 它的代价是所有任务竞争同一小段参数，遇到真正差异巨大的连续域时可能容量不足；虚拟离群点也只能近似旧决策边界。

### 6.5 PGP：池不增长，也可以保护更新方向

**一句话理解：** 不冻结整个旧 Prompt，也不为新任务加一套；而是只允许新梯度沿着“尽量不影响旧任务”的方向更新。[13]

**它修复的问题：** OVOR 式单 Prompt 虽然定长，却会被新任务反复修改；CODA 式冻结旧参数又必须扩容。PGP 尝试让同一池继续学习，同时约束更新方向。

先看图：黑线是普通 L2P 式前向，红线才是 PGP 新增的反向处理。下方三个椭圆表示输入空间、Prompt 空间以及二者组成的联合空间；SVD 从旧数据张成的方向中构造投影矩阵。

![PGP论文方法图：Prompt与Key梯度投影](/images/blog/visual-prompt-continual-learning-lineage/15-pgp-fig2.png)

*图源：PGP 原论文 Figure 2（PDF 第 4 页）。*

跟图走一遍：

1. 前向仍可使用 L2P 的 Query、Key、Top-K 和 Prompt 池，PGP 不是另造一个路由器。
2. 学完旧任务后，从旧输入、Prompt 或查询特征构造联合表示，用 SVD 找到旧知识主要占据的子空间。
3. 学新任务时先正常反向得到 Prompt 梯度 $g_P$。
4. 删除梯度中指向旧子空间的分量，只保留正交方向：

$$
g'_P=\Pi_{\mathcal S_{\mathrm{old}}^\perp}g_P.
$$

5. 对带 Key 的底座，Key 更新也会改变“猫图到底选哪段 Prompt”，因此 Prompt 梯度和 Key 梯度都要保护。

一个二维直觉：若横轴承载猫狗知识，就只准车辆梯度沿纵轴更新。它把抗遗忘从“冻结旧槽位”改成“限制共享参数的方向”，有机会保持固定 Prompt 形状。代价是需要保存或累计子空间统计；旧任务越多，受保护方向越多，剩余正交空间越小，新任务会逐渐学不动。

### 6.6 OS-Prompt：不用先跑完整 ViT，走到早层就检索

**一句话理解：** L2P 要先跑完一次 ViT 才知道选什么 Prompt；OS-Prompt 在同一次前向走到早期层时，立即用当时的 CLS 检索本层 Prompt，然后继续往后走。[14]

**它修复的问题：** Prompt 参数虽然少，标准 L2P/CODA 却为查询额外支付一整次 ViT 计算。模型越大，这笔开销越不能忽略。

先看图中粗实线的主干：图片只从左向右走一次。每到一层，当前 CLS 向上查询该层 Key-Prompt 池，再把组合 Prompt 放回本层。上方参考 ViT 只在训练时提供正则目标，推理时删除。

![OS-Prompt论文方法图：早层查询与一次前向](/images/blog/visual-prompt-continual-learning-lineage/14-os-prompt-fig3.png)

*图源：OS-Prompt 原论文 Figure 3（PDF 第 8 页）。*

跟图走一遍：

1. 图片进入冻结 ViT，先正常计算到第 $l$ 层，取得此时的 CLS：

$$
q_l=x_{l,[\mathrm{CLS}]},
$$

2. $q_l$ 与该层所有 Key 匹配，以连续权重组合 Prompt：

$$
\phi_l=\sum_m\cos(q_l,k_l^m)p_l^m.
$$

3. Prompt 随后进入本层或后续层。到下一层时再用新的中间 CLS 查询下一层池，整张图没有回到网络起点。

早层 CLS 的问题是“还没完全看懂图片”。OS-Prompt++ 因此在训练时额外使用参考 ViT 的最终 CLS，让早层 Query 对同一组 Key 的相似度分布模仿成熟 Query：

$$
\mathcal L_{\mathrm{QR}}=\sum_l\|A^l_{\mathrm{query}}-A^l_{\mathrm{ref}}\|_2^2.
$$

该正则只在训练时使用。推理一张猫图时，网络走到早层便选择偏向动物的 Prompt，然后一路完成六分类。论文报告训练和推理 GFLOPs 约降低 50%，而精度没有因省掉查询 ViT 明显下降。它解决的是**前向效率**，底层 Prompt 池本身的容量和遗忘问题仍取决于所采用的具体配置。

### 6.7 NoRGa：Prefix Prompt 本来就是注意力里的隐式 MoE

**一句话理解：** 每个 Prefix Value 都可以看成一个小专家，注意力分数就是它的门控；NoRGa 不换掉 ViT 的 FFN，只把 Prompt 专家的线性门控改成“线性残差 + 非线性”。[15]

**它修复的问题：** HiDe 已经把任务识别和分类校准拆开，但 Prompt 本身的注意力权重仍由简单线性点积决定，表达复杂任务关系时可能门控不足。

先看论文图。左半把原图像 Value 与 Prefix Value 竖着排在一起：它们都被注意力加权，所以都可看成专家。右半只对 Prompt 对应的分数块加非线性；原图像 Token 的注意力机制不变。

![NoRGa论文方法图：Prefix Prompt作为注意力专家](/images/blog/visual-prompt-continual-learning-lineage/10-norga-fig2.png)

*图源：NoRGa 原论文 Figure 2（PDF 第 5 页）。*

跟图走一遍：

1. 对某个 Query，普通自注意力给每个图像 Value 一个分数；加入 Prefix 后，还会给每个 Prompt Value 一个分数。
2. 若把 Value 当专家，Softmax 后的注意力权重就是 Router 的门控权重。
3. 原线性 Prompt 分数 $s$ 的表达能力有限，NoRGa 改为：

原始图像 Key/Value 可以看作 $N$ 个预训练专家；加入 $L$ 个 Prefix Key/Value，相当于再加入 $L$ 个可学习专家。对某个图像 Query，Prompt 专家 $j$ 的线性注意力分数记作 $s_{i,N+j}(X)$。

NoRGa 将其替换为：

$$
\widehat s_{i,N+j}(X)=s_{i,N+j}(X)+\alpha\,\sigma\bigl(\tau s_{i,N+j}(X)\bigr),
$$

   其中 $\alpha,\tau$ 是可学习标量，$\sigma$ 可用 `tanh` 等非线性。

4. 非线性项增强门控表达力；原始线性残差保留稳定梯度，避免纯非线性饱和。
5. 代码层面只需在注意力 Softmax 前修改 Prompt 对应分数块：

$$
A_{\mathrm{prompt}}\leftarrow A_{\mathrm{prompt}}+\alpha\sigma(\tau A_{\mathrm{prompt}}).
$$

在猫图上，不同 Patch Query 可以以非线性方式增强“动物轮廓 Prefix 专家”、抑制无关专家。但务必注意：**这不是每层七个 FFN 专家，也不是把视觉 Token 交给多个 MLP。** 原生 ViT FFN 完全保留，变化只发生在 Prefix 注意力分数上。

因此 NoRGa 与 HiDe 参数量几乎相同，提升来自更好的门控估计，而不是堆更多 Prompt；代价是它也继承 HiDe 的每任务 Prompt、两阶段推理和类别统计增长。

### 6.8 VPT-NSP²：CNN 的正交投影不能直接搬到 ViT

**一句话理解：** PGP 说“把梯度投到旧空间的正交方向”，VPT-NSP² 进一步问：ViT 注意力里究竟要对哪两个方向正交，才能真的保持旧输出？[16]

**它修复 PGP 的问题：** 普通线性层只需近似满足 $X\Delta W=0$；ViT 还有 Softmax 注意力、Query-Key 乘积、Value 聚合和 LayerNorm，直接照搬 CNN/线性层投影缺少结构保证。

论文图上方黑线是正常前向，蓝线是 Prompt 梯度反传；下方把原梯度投影后再更新。图只画出直观的一次投影，真正方法还包含注意力两侧的两个约束。

![VPT-NSP²论文方法图：ViT专用零空间投影](/images/blog/visual-prompt-continual-learning-lineage/11-vpt-nsp2-fig6.png)

*图源：VPT-NSP² 原论文 Figure 6（PDF 第 14 页）。*

跟图走一遍：

1. 学完猫狗和鸟类后，用它们的中间量估计哪些方向承载旧注意力关系。
2. 学车辆时正常计算候选 Prompt 更新 $P_G$。
3. 不直接使用 $P_G$。若新 Prompt 为 $P_{t+1}=P_t+\Delta P$，理想目标是在旧输入 $X_t$ 上保持 ViT 输出不变：

$$
f_{\mathrm{ViT}}(X_t\mid P_t)=f_{\mathrm{ViT}}(X_t\mid P_t+\Delta P).
$$

4. 为什么要左右夹？论文把自注意力拆成“亲和度计算”和“Value 聚合”，推导出两条充分条件：

$$
Q_{X_t}W_k^\top\Delta P^\top=0,
$$

$$
S_{P_t}\Delta P=0.
$$

第一条保护旧图像 Query 与 Prompt Key 形成的注意力关系，第二条保护旧 Prompt Value 的聚合结果。候选更新最终用左右两个近似零空间矩阵夹住：

$$
\Delta P=B_2P_GB_1.
$$

5. 同时约束 Prompt 均值和标准差漂移，近似满足 LayerNorm 所需的分布不变假设。

一个直觉是：左投影约束“旧图片怎样看 Prompt”，右投影约束“Prompt 被加权后输出什么”。相比 PGP，NSP² 的推进是根据 ViT 结构推导两个不同方向的保护。Prompt 主体可以固定，但要维护近似零空间的协方差或基；任务越多，过强保护仍会压缩新任务可学习方向，这正是 NSP²++ 后来修复的问题。

## 七、2025—2026：多条后续线汇合——聚合、生成、轻量化与共享

### 7.1 CAPrompt：不再先猜任务，而是循环聚合所有任务 Prompt

**一句话理解：** 不硬选一个任务 Prompt，而是先把所有历史 Prompt 加权混合；用第一次分类结果反过来更新权重，再混合、再分类，循环修正。[17]

**它修复的问题：** CPrompt 让分类器容忍选错 Prompt，却仍然存在一次离散选择。若猫图对猫狗任务和鸟类任务都略有相似，硬 Argmax 会把不确定性直接丢掉。CAPrompt 保留软权重。

论文图分训练与推理两部分。找出两个连续出现的“Prompt aggregation → prediction”：第一次从粗权重出发，分类结果产生新的任务概率；第二次用新概率重新聚合。外侧箭头表示推理时可继续循环。

![CAPrompt论文方法图：循环Prompt聚合](/images/blog/visual-prompt-continual-learning-lineage/18-caprompt-fig2.png)

*图源：CAPrompt 原论文 Figure 2（PDF 第 3 页）。*

跟图走一遍：

1. 已见任务各有 Prompt $\phi_1,\ldots,\phi_t$。先用均匀权重或无 Prompt 特征给出的粗任务概率做第一次混合：

$$
\bar\phi=\sum_{i=1}^{t}p_i\phi_i.
$$

2. 将 $\bar\phi$ 插入 ViT 做第一次全类别预测。把猫、狗两类概率相加就得到“任务 1 概率”，鸟类和车辆同理。
3. 用新的任务概率 $p_i$ 再聚合 Prompt，再执行第二次预测。
4. 若需要，继续沿图中循环箭头迭代，直到预设轮数。

论文训练时加入凹性与线性约束，让模型对 Prompt 插值的响应尽量平滑，使混合 Prompt 的理论行为更可控。

例如猫图第一轮可能给动物两个任务接近权重，分类后猫狗概率明显占优，第二轮便提高 $\phi_1$ 权重。它解决了任务误判的离散断点，却仍保存 $O(T)$ 个任务 Prompt；循环越多，推理主干计算越高。

### 7.2 CPG：让共享 MoE 生成器对旧样本输出不变

**一句话理解：** 用固定数量的 Prompt 专家和一个共享 Router 为每张图生成 Prompt；学新任务时同时保护 Router 与专家，使旧图片尽量仍生成原来的 Prompt。[18]

**它合并了两条前代路线：** DAP/EvoPrompt 负责“由输入生成”，PGP/NSP² 负责“共享参数更新不破坏旧响应”。与每任务新增专家不同，CPG 的 Router 和专家数固定。

论文图也分上下两半。上半是前向 MoE：输入特征→Router→Top-K 门控→Prompt 专家加权。下半是反向保护：Router 梯度与专家梯度分别投到旧任务子空间的正交方向。

![CPG论文方法图：固定MoE Prompt生成器与双投影](/images/blog/visual-prompt-continual-learning-lineage/19-cpg-fig1.png)

*图源：CPG 原论文 Figure 1（PDF 第 3 页）。*

跟图走一遍：

1. 每层预先放置固定数量的 Prompt 专家 $P_1,\ldots,P_M$，它们不是“任务 1 专家、任务 2 专家”的一一绑定。
2. 当前 CLS $x$ 输入共享 Router，得到专家分数；Top-K 后只混合少数专家：

$$
g=\operatorname{softmax}(xW_g),\qquad
a=\operatorname{softmax}(g+\Psi(g,K)),
$$

$$
\widehat P=aP.
$$

3. 例如猫图可以混合 70% 专家 2 和 30% 专家 5；卡车图可能混合专家 1 与专家 5。专家 5 因而可以承载共享“轮廓”知识。
4. 学车辆时，普通梯度会同时改变 Router 与专家。为让旧猫图仍走近似相同路由并得到近似相同 Prompt，论文推导两条保护条件：

$$
x^t\Delta W_g=0,\qquad a^t\Delta P=0.
$$

第一条让旧输入经过更新后的 Router 仍给出相近门控，第二条让旧门控组合更新后的专家仍产出相近 Prompt。它分别对旧输入协方差和旧门控协方差做 SVD，再投影两类梯度。

这是一条较干净的固定主体容量路线：不需要每任务 Router，也不需要每任务 Prompt。但“固定”不等于没有代价——要维护投影统计；任务越多，可安全更新的零空间越小，Router 或专家最终可能学不动。

### 7.3 Additive Prompt Tuning：连 Prompt Token 都不拼了

**一句话理解：** 不向序列拼接任何新 Token；每层只学两个向量，直接修改 CLS 的 Key 和 Value，再把新旧任务向量逐步融合。[19]

**它修复的问题：** VPT/L2P 拼接 Prompt 会拉长注意力序列，检索方法还可能需要第二次前向。OVOR 虽然单 Prompt，却仍使用常规 Prompt Token。Additive Prompt Tuning 把接口进一步压缩。

先看图左右差分。左边是旧式 Prompt 池与 Token 拼接；右边只有两个小向量分别加到 CLS 的 K/V 位置，Patch Token 数完全不变。最右侧箭头表示学完任务后的渐进融合。

![Additive Prompt Tuning论文方法图：从拼接改为加法](/images/blog/visual-prompt-continual-learning-lineage/20-additive-prompt-fig1.png)

*图源：Additive Prompt Tuning 原论文 Figure 1（PDF 第 4 页）。*

ICCV 2025 的 Additive Prompt Tuning 也简称 APT；不要与 2024 年的 AdaPromptCL 混淆。跟图走一遍：

1. 普通注意力先为 CLS 产生 $k_{\mathrm{CLS}}^l,v_{\mathrm{CLS}}^l$。
2. 第 $l$ 层只学习两个 $D$ 维向量，逐元素加到它们上：

$$
\widehat k_{\mathrm{CLS}}^l=k_{\mathrm{CLS}}^l+p_k^l,
\qquad
\widehat v_{\mathrm{CLS}}^l=v_{\mathrm{CLS}}^l+p_v^l.
$$

3. Patch 的 K/V 不改，也不增加序列长度；12 层 ViT 只需 24 个向量。
4. 没有 Prompt 池、Key、Top-K 或任务识别，一张图只走一次 ViT。

5. 学完新任务后，不永久保留新旧两套，而是用 Progressive Prompt Fusion 合成下一阶段唯一 Prompt：

$$
P_{t+1}^{\mathrm{PPF}}=\alpha P_t+(1-\alpha)P_{t+1}.
$$

在例子中，猫狗 Prompt 与鸟类 Prompt 被融合成动物阶段参数，再与车辆 Prompt 融合；推理永远只有一组。论文 ImageNet-R 成本表报告 16.80 GFLOPs、0.02M 可训练 Prompt 参数，L2P 为 35.73 GFLOPs、0.04M。它得到极简、定长、一次前向结构，代价是融合不可逆，所有历史知识必须压进极少向量。

### 7.4 MQMK：单查询、单 Key 太粗，那就做宽度搜索与深度搜索

**一句话理解：** 不再用一个通用 Query 与“每任务一个 Key”粗略匹配；每个任务 Prompt 都生成自己的 Query，再只与该任务内部的多个类别 Key 比较。[20]

**它修复的问题：** L2P/DualPrompt 的无 Prompt CLS 缺少任务专属视角，一个任务 Key 又把猫和狗压成同一个中心，复杂任务很难准确路由。

论文图要区分橙色训练线与黑色推理线。训练知道标签，直接选真任务 Prompt；推理不知道任务，才让每个历史任务 Prompt 分别生成 Query，再进行“任务内多 Key”匹配。

![MQMK论文方法图：多查询与多类别Key匹配](/images/blog/visual-prompt-continual-learning-lineage/21-mqmk-fig3.png)

*图源：MQMK 原论文 Figure 3（PDF 第 4 页）。*

跟图走一遍：

1. 每个任务保存 Prompt $P_t$，每个类别保存 Key $k_{t,j}$。任务 1 因此至少有猫 Key、狗 Key，而不是只有一个“动物任务 Key”。
2. 训练时标签告诉模型样本属于哪个任务，只用对应 Prompt，更新该 Prompt、局部类别 Key 和分类器。
3. 推理时把同一图片分别搭配 $P_1,P_2,\ldots,P_T$，为每个任务形成一个带该任务视角的 Query：

$$
Q_t=f(P_G;P_t;x)[\mathrm{CLS}],
$$

4. $Q_t$ 只与任务 $t$ 内部的类别 Key 比较，取局部 Top-K 相似度之和作为任务分数：

$$
S_t=\max_{\{c_m\}_{m=1}^{K}}\sum_{m=1}^{K}\cos(Q_t,k_{t,c_m}),
\qquad
\widehat t=\arg\max_t S_t.
$$

5. 哪个任务分数最高，就选择哪个任务 Prompt 完成最终分类。“多查询”做任务间宽度搜索，“多 Key”做任务内深度搜索。

一张猫图在任务 1 视角下会同时接近猫、狗 Key，而在车辆视角下不会；这种“局部密度证据”比与一个任务中心比较更可靠。论文报告困难场景下 Prompt 匹配率提升超过 30%。

但标准 MQMK 要为每个任务 Prompt 形成 Query：把 $T$ 份输入并入一个 Batch 只降低墙钟时间，不会消除总 FLOPs。高效版 MQMK-EI 先平均任务 Prompt，用一个融合 Query 近似，再两阶段匹配。它以更多计算和 $O(T+C)$ 的 Prompt/Key 换更准路由，是“精确检索”答案，不是固定容量答案。

### 7.5 VPT-NSP²++：保护太强，也是一种失败

**一句话理解：** VPT-NSP² 对所有受保护方向近似一视同仁；NSP²++ 先估计参数对旧任务和新任务分别有多重要，再决定哪些方向严守、哪些方向适度放行。[21]

**它修复前代的问题：** 零空间不是无限的。旧任务越多，被保护方向越多，剩余更新空间越小；模型虽然不忘旧知识，却可能再也学不动新知识。

这里必须说明一个来源边界：截至资料核验日期，TPAMI 正式全文为非开放获取，官方代码仓库没有附论文 PDF、海报或方法图，因此本文**不把上一节 VPT-NSP² 的 Figure 6 冒充成 NSP²++ 原图**。上一节原图只说明两者共享的“候选梯度→近似零空间投影→更新 Prompt”骨架；以下增量依据正式摘要与官方代码说明。

把上一节的投影视为一道阀门，NSP²++ 的变化是给阀门增加重要性感知：

1. 估计不同 Prompt 参数或方向对旧任务性能的重要性；重要者若被改动，更可能遗忘。
2. 同时估计它们对当前新任务的重要性；重要者若完全锁住，新任务更可能欠拟合。
3. “旧任务重要、新任务不重要”的方向强保护。
4. “新任务关键”的方向适当放松正交限制，让模型仍有可塑性。
5. 仍在近似零空间框架中更新，只是从统一强度变成参数重要性感知的强弱调节。

在例子中，动物知识占据的某些方向对车辆完全无用，可以锁紧；若“边缘与轮廓”方向对动物和车辆都重要，则不能一刀切封死，需要允许受控共享和重组。它说明固定容量路线不能只追求“旧输出一丝不变”，还必须管理保护与学习之间的预算。

### 7.6 Hash：每任务一套 Prompt 真的是最优吗？

**一句话理解：** 所有任务共享一个固定 Prompt 专家池，但每个任务保留自己的 Router；历史上越常被调用的专家，后续越少被选、越小幅更新，避免热门专家被写坏。[22]

**它修复的问题：** 每任务复制 Prompt 会线性增长；完全共享又会让少数热门 Prompt 被所有任务反复更新。Hash 希望保留共享带来的迁移，同时按真实使用历史保护专家。

先看论文主图：左侧展示选中 Prompt 如何插入 ViT 多层；中间是任务 Router 对固定专家池打分并 Top-K 混合；右侧是历史激活次数反馈成专家偏置。主图没有画测试前的任务识别，完整流程需结合补充材料补上。

![Hash论文方法图：共享Prompt专家池与历史感知路由](/images/blog/visual-prompt-continual-learning-lineage/17-hash-fig2.png)

*图源：Hash 原论文 Figure 2（PDF 第 4 页）；测试时 TII 选 Router 的步骤来自论文补充材料。*

跟图走训练阶段：

1. 全局只放固定数量的 Prompt 专家，不随任务增加。它们可以被猫狗、鸟类和车辆共同调用。
2. 当前训练任务 $t$ 使用 Router $R_t$ 给所有专家打分，取 Top-K 并归一化：

$$
\omega_k=\frac{e^{s_k}}{\sum_{j\in\operatorname{TopK}(s)}e^{s_j}},
\qquad
\widetilde P(x)=\sum_{k\in\operatorname{TopK}(s)}\omega_kP_k.
$$

3. 被选 Prompt 的加权和插入多个 Transformer Block。相关任务可共享专家，单个样本也可组合多个专家。
4. 系统累计每个专家过去被激活的次数 $H_e^t$。这不是类别，而是一份“专家使用账本”。
5. 历史次数一方面作为 Router 偏置，降低热门专家继续垄断选择的机会；另一方面缩小热门专家梯度：

$$
\widetilde g_e=\rho(H_e^t)g_e,qquad \rho'(H)\le 0.
$$

例如专家 2 已经频繁服务猫狗和鸟类，学车辆时就减少它的更新，让较少使用的专家承担更多新知识。这相当于按真实历史使用量保护，而不是见到旧专家就全部冻结。

再走一遍**完整测试流程**，这一点主图没有画全：

$$
x\xrightarrow{\text{无Prompt特征}}\text{TII预测}\hat t
\xrightarrow{\text{选 }R_{\hat t}}\text{Top-K共享专家}
\xrightarrow{\text{带Prompt ViT}}\hat y.
$$

也就是说，Hash 不是让所有任务 Router 都同时给专家打分。猫图先被 TII 判为任务 1，才调用 $R_1$；若 TII 错判成任务 3，就会调用 $R_3$。因此仍是两阶段推理，并继承任务识别误差。

Hash 真正固定的是**体积较大的 Prompt 专家池**；每个任务仍配置专属 Router，TII 输出维度随任务增长，还保存类别高斯统计。所以它比每任务复制完整 Prompt 更省，也在同设置表中很强，但还不是“所有状态都恒定”的最终答案。

## 八、把所有方法放回同一张设计表

| 方法 | Prompt 如何产生 | 如何处理未知任务 | 主干前向 | 主要历史状态 | 任务数增长 |
|---|---|---|---:|---|---|
| L2P | 固定池硬 Top-K | CLS 检索 Key | 2 | Prompt、Key、分类头 | Prompt 固定，头随类增长 |
| DualPrompt | 共享 G + 任务 E | 最近任务 Key | 2 | G、E、Key、头 | E/Key 为 $O(T)$ |
| S-Prompts | 每域 Prompt | 最近域中心 | 2 | Prompt、中心 | $O(T)$ |
| CODA | 组件连续加权 | 查询直接组合 | 2 | P/K/A 组件 | $O(T)$ 扩组件 |
| DAP | 实例生成 | 任务 Key 条件化 | 通常 2 | 生成器、任务 Key | Key 为 $O(T)$ |
| APG | 中层交叉注意力生成 | 输入条件生成 | 1 | 候选 Prompt、类统计 | 候选可扩，统计 $O(C)$ |
| HiDe | 每任务 Prompt | 显式 TII 头 | 2 阶段 | Prompt、两类高斯统计 | $O(T)+O(C)$ |
| EvoPrompt | 连续 FFN 记忆 | 无离散任务检索 | 通常 2 | 固定参考/工作记忆 | 最终 Prompt 记忆固定 |
| CPrompt | 每任务 Prompt | 多 Key 检索，训练容错 | 2 | Prompt、类 Key | $O(T)+O(C)$ |
| AdaPromptCL | 每语义组 Prompt | 组 Key 检索 | 通常 2 | Prompt 组、候选组 | $O(G)$，最坏 $O(T)$ |
| OVOR | 单 Prompt | 无须任务推断 | 1 | 单 Prompt、分类头 | Prompt 固定 |
| PGP | 继承底座 | 继承底座 | 依底座 | 子空间统计 | 可固定形状 |
| OS-Prompt | 中间 CLS 组合池 | 早层查询 | 1 | 继承底层池 | 依底座 |
| NoRGa | HiDe Prompt + 非线性门控 | HiDe 的 TII | 2 阶段 | 同 HiDe | $O(T)+O(C)$ |
| VPT-NSP²/++ | 单 Prompt 连续更新 | 无须任务推断 | 1 | Prompt、投影统计 | 主要状态固定形状 |
| CAPrompt | 所有任务 Prompt 循环加权 | 不做硬任务选择 | 多轮 | 任务 Prompt、分类头 | Prompt 为 $O(T)$ |
| CPG | 固定 MoE 生成 Prompt | Router 直接按实例组合 | 1 | 固定专家池、两类协方差 | 主体固定形状 |
| Additive PT | 每层两个向量修改 CLS 的 K/V | 无须任务推断 | 1 | 单组加法 Prompt | Prompt 固定 |
| MQMK | 每任务查询 + 每类 Key | 局部匹配后选任务 | 多查询或高效两阶段 | 任务 Prompt、类 Key | $O(T)+O(C)$ |
| Hash | 固定共享 Prompt 专家池 | HiDe 式 TII 选任务 Router | 2 阶段 | 固定池、Router、类统计、激活计数 | Prompt 固定，Router/TII $O(T)$，统计 $O(C)$ |

这张表透露出一个经常被忽略的事实：**很多高精度方法把难题从 Prompt 转移到了任务识别、类统计或 Router。** 所以“Prompt 参数只有 0.1%”不是完整资源结论。

## 九、谁最强？必须先问“在哪个协议里最强”

不能把不同论文的最佳数字直接拼成排行榜。预训练可能是 ImageNet-1K、监督 ImageNet-21K、DINO、iBOT 或 CLIP；数据划分、Epoch、分类头、随机种子和是否保存统计量也不同。

目前最有意义的一组横向数字来自 Hash 论文：它在同一 Sup-21K ViT-B/16、10 任务和同一训练管线中复现多种直接 Prompt-CIL 方法，最终平均准确率如下：

| 方法 | Split CIFAR-100 | Split ImageNet-R |
|---|---:|---:|
| L2P | 83.06 | 67.53 |
| DualPrompt | 87.30 | 70.93 |
| CODA-Prompt | 86.94 | 70.03 |
| CPrompt | 87.82 | 77.15 |
| HiDe-Prompt | 92.61 | 75.06 |
| NoRGa | 94.48 | 75.40 |
| EvoPrompt | 87.97 | 76.83 |
| OVOR | 85.99 | 76.11 |
| CPG | 90.63 | 78.63 |
| Hash | **95.02** | **79.02** |

因此，截至 2026 年 9 月，在我核验到的**同设置直接 Prompt-CIL 对比**中，Hash 给出了更强的这组结果；NoRGa 在 HiDe 体系和多种预训练范式下证据很稳；VPT-NSP²++ 则代表更新的固定 Prompt/长序列优化方向。三者回答的并不是同一个问题，不能用一句“谁是统一 SOTA”抹平。

如果需求不同，选择也不同：

- 要同设置精度：优先考察 Hash、NoRGa；
- 要固定 Prompt 容量与一次前向：考察 Additive Prompt Tuning、CPG、OVOR、VPT-NSP²++；
- 要避免离散检索错配：考察 EvoPrompt、APG；
- 要现有 L2P/CODA 减少一半主干计算：考察 OS-Prompt；
- 要任务流包含温和和剧烈语义变化：考察 AdaPromptCL；
- 要理解并修复训练—测试错配：CPrompt 最有启发性。

## 十、如果真要做一千个任务，应该从这条路线学到什么？

第一，**不要默认任务边界就是语义边界。** 随机类别切分时，给每个任务独立专家可能只是把基准的人工分组写进模型。

第二，**检索错误和分类错误必须分开测。** HiDe 的 WTP/TII/TAP、CPrompt 的两种一致性提供了比总准确率更好的诊断工具。

第三，**优先固定承载知识的大模块，再处理小状态。** 固定 Prompt 专家池、单 Prompt、连续 FFN 记忆都比每任务复制一套大 Prompt 更适合长流；之后再把每任务 Router 改成共享 Router，把全协方差改成有界原型库。

第四，**固定容量必然需要覆盖、融合或回收。** EvoPrompt 选择融合，OVOR 选择全部知识写入同一 Prompt，NSP² 选择限制更新方向，Hash 选择按使用历史保护共享专家。它们都没有创造无限容量，只是选择了不同的压缩代价。

第五，**一次前向很重要。** 当主干是 ViT-L 或更大模型时，Prompt 参数省下的内存可能远小于第二次主干前向带来的计算。真实部署应把 OS-Prompt 式中间查询或无检索单 Prompt 作为默认基线。

一个更完整的长期结构，可能是：

$$
\text{固定共享Prompt/低秩专家池}
+\text{一个共享Router}
+\text{Top-K稀疏执行}
+\text{历史使用保护}
+\text{有界原型库}
+\text{满载后的合并或驱逐策略}.
$$

这比“第 $t$ 个任务创建第 $t$ 个专家”难得多，却更接近真正的终身学习。

## 十一、Q&A：把最容易混淆的实现问题一次讲清

### Q1：L2P 的 Key 是手工特征，还是可学习向量？

是可学习向量。冻结 ViT 输出的 CLS 是查询，Key 自身通过匹配损失学习；Prompt 则通过分类损失学习。查询网络冻结，不等于 Key 冻结。

### Q2：标准 L2P 真的是两次 ViT，而不是论文示意图造成的误解？

是。官方 Google Research 代码的标准 CLS 配置先调用 `original_vit_model` 取得 `pre_logits`，再执行带 Prompt 的模型；训练和评估路径一致。

### Q3：那是不是池中每个 Prompt 都要处理所有视觉 Token？

不是。第一次无 Prompt 前向只产生查询；Top-K 以后，只有选中的 Prompt 进入第二次 ViT。Prompt 池检索本身只是归一化点积，远小于第二次主干计算。

### Q4：为什么不能先只处理 CLS，选中 Prompt 后再补算其余视觉 Token？

标准 ViT 的 CLS 从第一层开始就通过自注意力与所有 Patch Token 交互。没有先算好全部 Patch 的前层状态，就不能独立得到“正确的最终 CLS”。除非重新设计可缓存、可分段的主干，否则这不是简单的延迟执行。OS-Prompt 的做法正是安全地利用早层中间 CLS，而不是假装最终 CLS 可以脱离 Patch 单算。

### Q5：L2P 是否会每个任务增加一个 Prompt？

不会。L2P 的池大小是预设超参数，Prompt 可跨任务复用。每任务增加一套专属 Prompt 的典型方法是 DualPrompt、HiDe-Prompt、CPrompt 和 NoRGa。CODA 则每任务扩展一组组件。

### Q6：固定 Prompt 池是不是就解决了数千任务增长？

只解决参数增长，不保证容量足够。固定池可能出现槽位竞争、Key 冲突和旧知识覆盖；而分类头通常仍随累计类别数增长。真正审计还要看 Router、类统计和计算。

### Q7：NoRGa 是不是每层把 ViT 的 FFN 换成了一个 MoE？

不是。NoRGa 改的是自注意力中的 Prefix Prompt 分数，不是 ViT Block 后半段的 MLP/FFN。它把 Prefix Key/Value 解释成注意力里的新增专家，再给这些 Prompt 专家的线性分数加非线性残差项。原生 FFN 仍然存在。

### Q8：NoRGa 为什么参数量几乎不增加？

因为新增核心只有每处门控的少量标量 $\alpha,\tau$；Prompt 本身来自 HiDe-Prompt。它提升的是 Prompt 专家门控的表达力和参数估计效率。不过 HiDe 的任务 Prompt 池仍随任务增长。

### Q9：为什么 OVOR 的一个 Prompt 反而可能胜过 Prompt 池？

在随机类增量中，任务分组往往没有独立语义。多 Prompt 会额外引入任务识别错误和跨头偏置；共享一个 Prompt 避免了路由错误，而冻结大主干又限制了表示漂移。此时主要矛盾可能是分类边界，不是特征专家不足。

### Q10：固定容量方法能保证无限任务零遗忘吗？

不能。对无限多、彼此独立的知识，不可能同时要求零回放、零扩容、零遗忘和任意精度。固定容量方法只能通过共享、压缩、覆盖、融合、投影或驱逐决定“哪些知识值得保留”。

## 结语

L2P 最重要的遗产，不是某一个 Prompt 池公式，而是把持续学习从“怎样保护整网权重”改写成“怎样管理一个小型可寻址记忆系统”。

后续四年里，这个记忆系统先被拆成共享与专属 Prompt，又被改造成连续组件、实例生成器和 FFN；研究者随后发现，真正的瓶颈还包括任务识别、全局分类头、双前向和参数隔离；于是又出现了训练容错、单 Prompt、零空间投影、非线性残差门控和历史感知共享池。

所以这条路线最值得记住的不是方法名，而是六次认知升级：

1. Prompt 可以像 RAG 一样被内容寻址；
2. 共享知识与专属知识需要不同载体；
3. 检索、分类和任务推断是三个不同误差源；
4. Prompt 可以是连续函数，而不只是池中的一行参数；
5. 参数少不等于前向便宜，也不等于长期容量有界；
6. 真正可扩展的方向不是无限隔离，而是共享、保护、融合和有代价的回收。

这也解释了为什么今天再看 L2P，它既不是过时的起点，也不是最终答案。它更像一块试金石：后来的每篇论文，都在回答 L2P 当年故意留下的某个问题。

## 参考资料

[0] Jia et al.，Visual Prompt Tuning，ECCV 2022：[论文](https://arxiv.org/abs/2203.12119)

[1] Wang et al.，Learning to Prompt for Continual Learning，CVPR 2022：[论文与代码入口](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_Learning_To_Prompt_for_Continual_Learning_CVPR_2022_paper.html)

[2] Wang et al.，DualPrompt: Complementary Prompting for Rehearsal-free Continual Learning，ECCV 2022：[论文](https://arxiv.org/abs/2204.04799)

[3] Smith et al.，CODA-Prompt，CVPR 2023：[论文与补充材料](https://openaccess.thecvf.com/content/CVPR2023/html/Smith_CODA-Prompt_COntinual_Decomposed_Attention-Based_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2023_paper.html)

[4] Wang et al.，S-Prompts Learning with Pre-trained Transformers，NeurIPS 2022：[论文](https://proceedings.neurips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html)

[5] Jung et al.，Generating Instance-level Prompts for Rehearsal-free Continual Learning，ICCV 2023：[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Jung_Generating_Instance-level_Prompts_for_Rehearsal-free_Continual_Learning_ICCV_2023_paper.html)

[6] Tang et al.，When Prompt-based Incremental Learning Does Not Meet Strong Pretraining，ICCV 2023：[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Tang_When_Prompt-based_Incremental_Learning_Does_Not_Meet_Strong_Pretraining_ICCV_2023_paper.html)

[7] Gao et al.，A Unified Continual Learning Framework with General Parameter-Efficient Tuning，ICCV 2023：[论文](https://openaccess.thecvf.com/content/ICCV2023/html/Gao_A_Unified_Continual_Learning_Framework_with_General_Parameter-Efficient_Tuning_ICCV_2023_paper.html)

[8] Wang et al.，Hierarchical Decomposition of Prompt-Based Continual Learning，NeurIPS 2023：[论文](https://proceedings.neurips.cc/paper_files/paper/2023/hash/d9f8b5abc8e0926539ecbb492af7b2f1-Abstract-Conference.html)

[9] Wang et al.，Evolving Parameterized Prompt Memory for Continual Learning，AAAI 2024：[论文](https://doi.org/10.1609/aaai.v38i12.29231)

[10] Gao et al.，Consistent Prompting for Rehearsal-Free Continual Learning，CVPR 2024：[论文](https://openaccess.thecvf.com/content/CVPR2024/html/Gao_Consistent_Prompting_for_Rehearsal-Free_Continual_Learning_CVPR_2024_paper.html)

[11] Kim et al.，One Size Fits All for Semantic Shifts: Adaptive Prompt Tuning for Continual Learning，ICML 2024：[论文](https://proceedings.mlr.press/v235/kim24ai.html)

[12] Li et al.，OnePrompt with Virtual Outlier Regularization for Rehearsal-Free Class-Incremental Learning，ICLR 2024：[论文](https://arxiv.org/abs/2402.04129)

[13] Qiao et al.，Prompt Gradient Projection for Continual Learning，ICLR 2024：[论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/46c2a9a6f2b2be68682013eb1173c801-Abstract-Conference.html)

[14] Kim et al.，One-stage Prompt-based Continual Learning，ECCV 2024：[论文与补充材料](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/2053_ECCV_2024_paper.php)

[15] Doan et al.，Mixture of Experts Meets Prompt-Based Continual Learning，NeurIPS 2024：[论文](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d78d68cae595fabadd187b583ee8708e-Abstract-Conference.html)

[16] Lu et al.，Visual Prompt Tuning in Null Space for Continual Learning，NeurIPS 2024：[论文](https://papers.neurips.cc/paper_files/paper/2024/hash/0f06be0008bc568c88d76206aa17954f-Abstract-Conference.html)

[17] Li and Zhou，CAPrompt: Cyclic Prompt Aggregation for Pre-Trained Model Based Class Incremental Learning，AAAI 2025：[论文](https://ojs.aaai.org/index.php/AAAI/article/view/34027)

[18] Lu et al.，Training Consistent Mixture-of-Experts-Based Prompt Generator for Continual Learning，AAAI 2025：[论文](https://ojs.aaai.org/index.php/AAAI/article/view/34108)

[19] Chen et al.，Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning，ICCV 2025：[论文](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Achieving_More_with_Less_Additive_Prompt_Tuning_for_Rehearsal-Free_Class-Incremental_ICCV_2025_paper.html)

[20] Tu et al.，Multiple Queries with Multiple Keys: A Precise Prompt Matching Paradigm for Prompt-based Continual Learning，ACM MM 2025：[论文](https://arxiv.org/abs/2501.12635)

[21] Zhang et al.，VPT-NSP²++: Importance-Aware Visual Prompt Tuning in Null Space for Continual Learning，TPAMI 2025 在线发表、2026 卷期：[论文信息](https://doi.org/10.1109/TPAMI.2025.3642298)；[官方代码](https://github.com/zugexiaodui/VPT-NSP2pp)

[22] Li et al.，Is Parameter Isolation Better for Prompt-Based Continual Learning?，CVPR 2026：[论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Is_Parameter_Isolation_Better_for_Prompt-Based_Continual_Learning_CVPR_2026_paper.html)

---

资料核验截至 2026 年 9 月。文中实验数字只在明确注明的同一复现表内横向比较；不同预训练、数据切分与训练协议的结果没有混成统一排行榜。
