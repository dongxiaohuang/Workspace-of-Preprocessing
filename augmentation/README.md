<!-- TOC -->

1. [Data-Augmentation](#data-augmentation)
2. [Invotation](#invotation)
3. [Sample consistency test](#sample-consistency-test)
    1. [判断两个样本集的分布是否一致](#判断两个样本集的分布是否一致)
    2. [判断生成后的数据与原始数据的分布是否一致](#判断生成后的数据与原始数据的分布是否一致)
4. [Solutions for Preprocessing Data Augemnt](#solutions-for-preprocessing-data-augemnt)
    1. [语义相似词替换](#语义相似词替换)
        1. [BERT](#bert)
        2. [ERNIE](#ernie)
        3. [BERT-WWW](#bert-www)
        4. [synonyms](#synonyms)
    2. [Random Insertion](#random-insertion)
    3. [Random Swap](#random-swap)
    4. [Random  Deletion](#random--deletion)
    5. [回译(相对好用)](#回译相对好用)
    6. [文档剪辑（长文本）](#文档剪辑长文本)
    7. [文本生成](#文本生成)
    8. [预训练语言模型](#预训练语言模型)
    9. [文本更正](#文本更正)
    10. [基于上下文的数据增强](#基于上下文的数据增强)
    11. [扩句-缩句-句法](#扩句-缩句-句法)
    12. [HMM-marko（质量较差）](#hmm-marko质量较差)
5. [Reference](#reference)
    1. [Links](#links)
    2. [Tools](#tools)

<!-- /TOC -->


<a id="markdown-data-augmentation" name="data-augmentation"></a>
# Data-Augmentation

<a id="markdown-invotation" name="invotation"></a>
# Invotation

- NLP领域的数据属于离散数据, 小的扰动会改变含义，因此可以用数据增强的方式增强模型的泛化能力

- 常见的问题
  - Imbalance Label
  - Corrupter Label
  - Imbalance Diversity (数据多样性不平衡)


<a id="markdown-sample-consistency-test" name="sample-consistency-test"></a>
# Sample consistency test

<a id="markdown-判断两个样本集的分布是否一致" name="判断两个样本集的分布是否一致"></a>
## 判断两个样本集的分布是否一致

<a id="markdown-判断生成后的数据与原始数据的分布是否一致" name="判断生成后的数据与原始数据的分布是否一致"></a>
## 判断生成后的数据与原始数据的分布是否一致

<a id="markdown-solutions-for-preprocessing-data-augemnt" name="solutions-for-preprocessing-data-augemnt"></a>
# Solutions for Preprocessing Data Augemnt


<a id="markdown-语义相似词替换" name="语义相似词替换"></a>
## 语义相似词替换

- 随机的选一些词并用它们的同义词来替换这些词, 例如，将句子“我非常喜欢这部电影”改为“我非常喜欢这个影片”，这样句子仍具有相同的含义，很有可能具有相同的标签
- 但这种方法可能没什么用，因为同义词具有非常相似的词向量，因此模型会将这两个句子当作相同的句子，而在实际上并没有对数据集进行扩充。

<a id="markdown-bert" name="bert"></a>
### BERT

<a id="markdown-ernie" name="ernie"></a>
### ERNIE

<a id="markdown-bert-www" name="bert-www"></a>
### BERT-WWW

<a id="markdown-synonyms" name="synonyms"></a>
### synonyms
 

<a id="markdown-random-insertion" name="random-insertion"></a>
## Random Insertion

<a id="markdown-random-swap" name="random-swap"></a>
## Random Swap 

- code：<https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py>

<a id="markdown-random--deletion" name="random--deletion"></a>
## Random  Deletion

- code：<https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py>

<a id="markdown-回译相对好用" name="回译相对好用"></a>
## 回译(相对好用)

- 用机器翻译把一段英语翻译成另一种语言，然后再翻译回英语。
- 已经成功的被用在Kaggle恶意评论分类竞赛中
- 反向翻译是NLP在机器翻译中经常使用的一个数据增强的方法， 其本质就是快速产生一些不那么准确的翻译结果达到增加数据的目的
- 例如，如果我们把“I like this movie very much”翻译成俄语，就会得到“Мне очень нравится этот фильм”，当我们再译回英语就会得到“I really like this movie” ，回译的方法不仅有类似同义词替换的能力，它还具有在保持原意的前提下增加或移除单词并重新组织句子的能力
- 回译可使用python translate包和textblob包（少量翻译）

"""
1.在线翻译工具（中文->[英、法、德、俄、西班牙、葡萄牙、日、韩、荷兰、阿拉伯]等语言）        - 谷歌翻译(google)，谷歌翻译不用说，应该是挺好的，语言支持最多，不过我暂时还不会翻墙注册账户       - 百度翻译(baidu)，百度翻译不用说，国内支持翻译语言最多的了(28种互译)，而且最大方了，注册账户后每月有200万字符的流量，大约是2M吧，超出则49元人民币/百万字符       - 有道翻译(youdao)，初始接触网络的时候我最喜欢用有道翻译了，但死贵，只有100元体验金，差评。才支持11种语言，48元/百万字符       - 搜狗翻译(sougou)，对于搜狗印象还行吧，毕竟是能做搜索引擎的公司嘛。78种语言，200元体验金，常见语言40元/百万字符,非常见语言60元/百万字符       - 腾讯翻译(tencent)，总觉得腾讯AI是后知后觉了，公司调用腾讯接口老是变来变去的，这次也是被它的sign加密给恶心到了，空格改为+。或许对企鹅而言，人工智能不那么重要吧。                          -有两个，一个是翻译君一个是AIlab什么的，支持的语言少些。似乎还在开发中，不限额不保证并发，php开发没有python的demo       - 必应翻译(bing)，微软的东西，你懂的，没有尝试，直接在网页上试试还可以吧       - 可以采用工具、模拟访问网页、或者是注册账号等    - 2.离线翻译工具       - 1.自己写，收集些语料，seq2seq,nmt,transformer       - 2.小牛翻译，比较古老的版本了，win10或者linux都可以，不过只有训练好的中英互译             地址:小牛翻译开源社区
"""

- 或者使用百度翻译或谷歌翻译的api通过python实现
- 参考：https://github.com/dupanfei1/deeplearning-util/tree/master/nlp
- APIs
  - mtranslate

<a id="markdown-文档剪辑长文本" name="文档剪辑长文本"></a>
## 文档剪辑（长文本）

- 新闻文章通常很长，在查看数据时，对于分类来说并不需要整篇文章。 文章的主要想法通常会重复出现。将文章裁剪为几个子文章来实现数据增强，这样将获得更多的数据

<a id="markdown-文本生成" name="文本生成"></a>
## 文本生成

- 生成文本
- [Data Augemtation Generative Adversarial Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.04340)
- [Triple Generative Adversarial Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.02291)
- [Semi-Supervised QA with Generative Domain-Adaptive Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.02206)

<a id="markdown-预训练语言模型" name="预训练语言模型"></a>
## 预训练语言模型

- ULMFIT
- Open-AI transformer 
- BERT

<a id="markdown-文本更正" name="文本更正"></a>
## 文本更正

- 中文如果是正常的文本多数都不涉及，但是很多恶意的文本，里面会有大量非法字符，比如在正常的词语中间插入特殊符号，倒序，全半角等。还有一些奇怪的字符，就可能需要你自己维护一个转换表了
- 文本泛化
- 表情符号、数字、人名、地址、网址、命名实体等，用关键字替代就行。这个视具体的任务，可能还得往下细化。比如数字也分很多种，普通数字，手机号码，座机号码，热线号码、银行卡号，QQ号，微信号，金钱，距离等等，并且很多任务中，这些还可以单独作为一维特征。还得考虑中文数字阿拉伯数字等
- 中文将字转换成拼音，许多恶意文本中会用同音字替代
- 如果是英文的，那可能还得做词干提取、形态还原等，比如fucking,fucked -> fuck
- 去停用词

<a id="markdown-基于上下文的数据增强" name="基于上下文的数据增强"></a>
## 基于上下文的数据增强

- 方法论文：Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relations
- 方法实现代码：使用双向循环神经网络进行数据增强
- 该方法目前针对于英文数据进行增强，实验工具：spacy（NLP自然语言工具包）和chainer（深度学习框架）

<a id="markdown-扩句-缩句-句法" name="扩句-缩句-句法"></a>
## 扩句-缩句-句法
- 1.句子缩写，查找句子主谓宾等
- 有个java的项目，调用斯坦福分词工具(不爱用)，查找主谓宾的        
- 地址为:（主谓宾提取器）hankcs/MainPartExtractor    
- 2.句子扩写     
- 3.句法

<a id="markdown-hmm-marko质量较差" name="hmm-marko质量较差"></a>
## HMM-marko（质量较差）
- HMM生成句子原理: 根据语料构建状态转移矩阵，jieba等提取关键词开头，生成句子
- 参考项目:takeToDreamLand/SentenceGenerate_byMarkov


<a id="markdown-reference" name="reference"></a>
# Reference

<a id="markdown-links" name="links"></a>
## Links

- https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/
- https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
- 知乎:中文自然语言处理中的数据增强方式
    - https://www.zhihu.com/question/305256736

<a id="markdown-tools" name="tools"></a>
## Tools

- SMOTE
- imblance learn
- scikit learn
- synonyms
