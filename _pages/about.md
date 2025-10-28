---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I received my Ph.D. degree at the School of Information and Electronics & School of Computer Science and Technology, Beijing Institute of Technology (BIT), supervised by Prof. <a href="https://ying-fu.github.io/">Ying Fu</a>. 
I am also fortunate to collaborate closely with <a href="https://zhouhy.org/">Hong-yu Zhou</a> and <a href="https://sites.google.com/view/linguedu/home">Lin Gu</a> (RIKEN AIP / The University of Tokyo). 
I received my M.S. degree from BIT in 2021 and my B.S. degree from China University of Geosciences (Beijing) in 2019. 

My research interests primarily lie in computer vision and deep learning, including:
- Frequency Learning
- Image Segmentation, Object Detection
- Low-light Image Enhancement and Recognition
- Remote Sensing Image Processing
- Representation Learning for Transformers & CNNs
- Medical Large Language Models

You can find my full publication list on my <a href='https://scholar.google.com/citations?user=EGlOtL4AAAAJ'>Google Scholar</a> profile.
<!-- <a href='https://scholar.google.com/citations?user=EGlOtL4AAAAJ'><img src="https://img.shields.io/google-scholar/citations/EGlOtL4AAAAJ?style=flat&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&label=citations"></a>. -->

I am open for collaborations in research, especially in the fields of Image Understanding, Representation Learning and Medical LLMs. Please don’t hesitate to get in touch if my research interests you.

I am actively seeking positions in both academia and industry, starting in the Fall of 2025.

<span class='anchor' id='-cv'></span>
**[Download my CV here](./_pages/Linwei_Chen_CV.pdf)**

<span class='anchor' id='-education'></span>

# 🎓 Education
- *2021.09 - 2025.06*, <a href="https://english.bit.edu.cn/"><img class="svg" src="/images/BIT_logo.png" width="20pt"></a> Beijing Institute of Technology, Ph.D. in Electronic Information
- *2019.09 - 2021.06*, <a href="https://english.bit.edu.cn/"><img class="svg" src="/images/BIT_logo.png" width="20pt"></a> Beijing Institute of Technology, M.S. in Software Engineering
- *2014.09 - 2019.06*, <a href="https://en.cugb.edu.cn/"><img class="svg" src="/images/CUGB_logo.svg" width="20pt"></a> China University of Geosciences (Beijing), B.S. in Mechanical Engineering

<span class='anchor' id='-publications'></span>
# 📝 Selected Publications

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICCV 2025</div><img src='./images/ICCV2025-FDAM.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Frequency-Dynamic Attention Modulation for Dense Prediction**  
**Linwei Chen**, Lin Gu, Ying Fu\*.  
*International Conference on Computer Vision (ICCV)*, 2025.  
[[PDF]](https://arxiv.org/abs/2507.12006) <a href='https://github.com/Linwei-Chen/FDAM'><img src="https://img.shields.io/github/stars/Linwei-Chen/FDAM?style=social" alt="Stars"></a> [[机器之心 Report]](https://mp.weixin.qq.com/s/eCN-0K5TQCX0nWA0CacWTQ) [[BibTeX]](./bibtex/ICCV2025-FDAM.bib)

🧐 *Revealing a “frequency vanishing” phenomenon in ViTs and introducing a frequency-dynamic attention mechanism that strengthens feature representations for dense prediction.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">TPAMI 2025</div><img src='./images/TPAMI2025-SFM.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Spatial Frequency Modulation for Semantic Segmentation**  
**Linwei Chen**, Ying Fu\*, Lin Gu, Dezhi Zheng, Jifeng Dai.  
*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2025.  
[[PDF]](https://arxiv.org/abs/2507.11893) [[IEEE]](https://ieeexplore.ieee.org/document/11095993) [[公众号]](https://mp.weixin.qq.com/s/Q50ViJ22Gikb3CfI_LDIFw) <a href='https://github.com/Linwei-Chen/SFM'><img src="https://img.shields.io/github/stars/Linwei-Chen/SFM?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/TPAMI2025-SFM.bib)

😎 *Proposing **Spatial Frequency Modulation (SFM)** to mitigate the degradation of high-frequency representations during widely-used downsampling.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2025</div><img src='images/FDConv2025cvpr.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Frequency Dynamic Convolution for Dense Image Prediction**  
**Linwei Chen**, Lin Gu, Liang Li, Chenggang Yan, Ying Fu\*.  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2025.  
[[PDF]](https://arxiv.org/abs/2503.18783) [[Zhihu 知乎]](https://zhuanlan.zhihu.com/p/32726185863) <a href='https://github.com/Linwei-Chen/FDConv'><img src="https://img.shields.io/github/stars/Linwei-Chen/FDConv?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/fuying-2025-CVPR-chenlinwei.txt)

🤭 *Presenting a parameter-efficient **frequency dynamic convolution (FDConv)** operating in the frequency domain, enabling 64× kernels without increasing parameter cost.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">TPAMI 2024</div><img src='images/TPAMI2024chenlinwei.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Frequency-aware Feature Fusion for Dense Image Prediction**  
**Linwei Chen**, Ying Fu\*, Lin Gu, Chenggang Yan, Tatsuya Harada, Gao Huang.  
*IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)*, 2024. 
(**<span style="color:red">[WOS Citation Top 0.3%](./images/TPAMI2024-FreqFusion-WOS.png)</span>**, **<a href="http://www.ic-icc.cn/2025/Highlight_Paper_Speakers.php"><span style="color:red">ICIC2025 Invited Hilight Paper</span></a>**)  
[[PDF]](https://www.arxiv.org/abs/2408.12879) [[Zhihu 知乎]](https://zhuanlan.zhihu.com/p/20512332940) <a href='https://github.com/Linwei-Chen/FreqFusion'><img src="https://img.shields.io/github/stars/Linwei-Chen/FreqFusion?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/fuying-2024-TPAMI-chenlinwei.txt)

🥳 *Tackling feature fusion by analyzing frequency characteristics, proposing a plug&play method **FrequFusion** to effectively combine features for dense prediction.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">CVPR 2024</div><img src='images/FADC2024cvpr.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Frequency-Adaptive Dilated Convolution for Semantic Segmentation**  
**Linwei Chen**, Lin Gu, Ying Fu\*.  
*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2024. 
(**<span style="color:red">Highlight Poster, Acceptance Rate 2.8%</span>**, **<span style="color:red">[WOS Citation Top 0.2%](./images/CVPR20224-FADC-WOS.png)</span>**)  
[[PDF]](https://arxiv.org/abs/2403.05369) [[Zhihu 知乎]](https://zhuanlan.zhihu.com/p/719255929) <a href='https://github.com/Linwei-Chen/FADC'><img src="https://img.shields.io/github/stars/Linwei-Chen/FADC?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/fuying-2024-CVPR-chenlinwei.txt)

🤓 *Introducing **Frequency Adaptive Dilated Convolution (FADC)** that adaptively sets its dilation rate from frequency cues, intelligently expanding the receptive field while preserving high-frequency details.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">ICLR 2024</div><img src='images/ICLR2024chenlinwei.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**When Semantic Segmentation Meets Frequency Aliasing**  
**Linwei Chen**, Lin Gu, Ying Fu\*.  
*International Conference on Learning Representations (ICLR)*, 2024.  
[[PDF]](https://openreview.net/pdf?id=SYBdkHcXXK) [[Zhihu 知乎]](https://zhuanlan.zhihu.com/p/678596052) <a href='https://github.com/Linwei-Chen/Seg-Aliasing'><img src="https://img.shields.io/github/stars/Linwei-Chen/Seg-Aliasing?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/fuying-2024-ICLR-chenlinwei.txt)

😆 *Providing the first systematic study of the frequency aliasing and hard pixel problem in semantic segmentation and proposes a simple yet effective solution.*

</div>
</div>

<div class='paper-box'><div class='paper-box-image'><div><div class="badge">IJCV 2023</div><img src='images/IJCV2023chenlinwei.png' alt="sym" width="100%"></div></div>
<div class='paper-box-text' markdown="1">

**Instance Segmentation in the Dark**  
**Linwei Chen**, Ying Fu\*, Kaixuan Wei, Dezhi Zheng, Felix Heide.  
*International Journal of Computer Vision (IJCV)*, 2023.

(**<span style="color:red">[WOS Citation Top 3.8%](./images/IJCV2023-segindark-WOS.png)</span>**)  

[[PDF]](https://link.springer.com/article/10.1007/s11263-023-01808-8) [[Zhihu 知乎]](https://zhuanlan.zhihu.com/p/656570195) <a href='https://github.com/Linwei-Chen/LIS'><img src="https://img.shields.io/github/stars/Linwei-Chen/LIS?style=social" alt="Stars"></a> [[BibTeX]](./bibtex/fuying-2023-IJCV-chenlinwei.txt)

🌚 *Tackling the challenging task of instance segmentation in extreme low-light conditions by introducing a new dataset **Low-light Instance Segmentation (LIS)** and a specialized method.*


</div>
</div>

<span class='anchor' id='-talks'></span>
# 🎤 Invited Talks
- *[2025.08.22]* "基于频域分析的视觉表征学习与场景理解" (Visual Representation Learning & Scene Understanding in the Frequency Domain), *CCF Forum on Machine Embodied Interaction Intelligence*. [[Slides](./pdf/基于频域分析的视觉表征学习与场景理解-v5.4-4.pdf)] [[Event](https://hhme.ccf.org.cn/forum10.html)]
- *[2025.07.28]* "Frequency-aware Feature Fusion for Dense Image Prediction", *Invited Highlight Paper Speaker at the International Conference on Intelligent Computing (ICIC)*. [[Slides](./pdf/FreqFusion-ICIC_ENG.pdf)] [[Event](http://www.ic-icc.cn/2025/Highlight_Paper_Speakers.php)]
- *[2025.05.10]* "从图像生成到CT图像增强应用" (From Image Generation to CT Image Enhancement Application), *1st Academic Conference on Innovative Development of Smart Health and Intelligent Healthcare (IDAC)*. [[Slides](./pdf/从图像生成到CT图像增强应用.pdf)] [[Event](https://idac2025.sciconf.cn/)]

<span class='anchor' id='-awards'></span>
# 🏅 Awards and Honors
- *2024* China National Scholarship for Doctoral Students, Ministry of Education of the People’s Republic of China 
- *2024* CVPR 2024 PBDL International Challenge, Low-Light Track: **2nd Place** (Segmentation), **3rd Place** (Detection)
- *2024* BMVC 2024 Outstanding Reviewer
- *2022* Beijing Institute of Technology Doctoral Special Scholarship
- *2021* Rocket Force "Intelligent Arrow - Fire Eye" AI Challenge, 7th Place Nationally (Excellence Award)

<span class='anchor' id='-professional-service'></span>
# 💬 Professional Service

### Journal Reviewer
- IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
- IEEE Transactions on Image Processing (TIP)
- International Journal of Computer Vision (IJCV)
- ISPRS Journal of Photogrammetry and Remote Sensing (ISPRS)
- Pattern Recognition (PR)

### Conference Reviewer
- IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) 2023, 2024, 2025
- IEEE/CVF International Conference on Computer Vision (ICCV) 2025
- European Conference on Computer Vision (ECCV) 2024
- AAAI Conference on Artificial Intelligence (AAAI) 2025, 2026
- Conference on Neural Information Processing Systems (NeurIPS) 2025, 2026
- British Machine Vision Conference (BMVC) 2023, 2024, 2025
- ACM Multimedia Conference (MM) 2025
- IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) 2026

<span class='anchor' id='-teaching'></span>
# 🧑‍🏫 Teaching
- Teaching Assistant, *Computer Vision*, Beijing Institute of Technology, 2021 - 2024 Fall.

<span class='anchor' id='-patents'></span>
# 📜 Patents
- I have 8 pending national patents, with 3 granted.