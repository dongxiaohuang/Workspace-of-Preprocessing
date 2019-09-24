# Data-Augmentation

## Reference

+ https://www.reddit.com/r/MachineLearning/comments/12evgi/classification_when_80_of_my_training_set_is_of/
+ https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/

## NLP

+ 离散
+ 小的扰动会改变含义

## Trick of Augmentation

+ confusion matrix

  + 观察混淆矩阵，找到需要重点增强的类别

+ random drop and shuffle
  + 提升数据量
  + code：<https://github.com/dupanfei1/deeplearning-util/blob/master/nlp/augment.py>

+ 同义词替换
  + 随机的选一些词并用它们的同义词来替换这些词
  + 例如，将句子“我非常喜欢这部电影”改为“我非常喜欢这个影片”，这样句子仍具有相同的含义，很有可能具有相同的标签
  + 但这种方法可能没什么用，因为同义词具有非常相似的词向量，因此模型会将这两个句子当作相同的句子，而在实际上并没有对数据集进行扩充。

+ 回译

  + 在这个方法中，用机器翻译把一段英语翻译成另一种语言，然后再翻译回英语。
  + 这个方法已经成功的被用在Kaggle恶意评论分类竞赛中
  + 反向翻译是NLP在机器翻译中经常使用的一个数据增强的方法， 其本质就是快速产生一些不那么准确的翻译结果达到增加数据的目的
  + 例如，如果我们把“I like this movie very much”翻译成俄语，就会得到“Мне очень нравится этот фильм”，当我们再译回英语就会得到“I really like this movie” ，回译的方法不仅有类似同义词替换的能力，它还具有在保持原意的前提下增加或移除单词并重新组织句子的能力
  + 回译可使用python translate包和textblob包（少量翻译）
  + 或者使用百度翻译或谷歌翻译的api通过python实现
  + 参考：https://github.com/dupanfei1/deeplearning-util/tree/master/nlp
  + APIs
    + mtranslate

+ 文档剪辑（长文本）

  + 新闻文章通常很长，在查看数据时，对于分类来说并不需要整篇文章。 文章的主要想法通常会重复出现。将文章裁剪为几个子文章来实现数据增强，这样将获得更多的数据

+ GAN

  + 生成文本
  + [Data Augemtation Generative Adversarial Networks](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1711.04340)
  + [Triple Generative Adversarial Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1703.02291)
  + [Semi-Supervised QA with Generative Domain-Adaptive Nets](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1702.02206)

+ 预训练语言模型

  + ULMFIT
  + Open-AI transformer 
  + BERT

+ 文本更正

  + 中文如果是正常的文本多数都不涉及，但是很多恶意的文本，里面会有大量非法字符，比如在正常的词语中间插入特殊符号，倒序，全半角等。还有一些奇怪的字符，就可能需要你自己维护一个转换表了

+ 文本泛化

  + 表情符号、数字、人名、地址、网址、命名实体等，用关键字替代就行。这个视具体的任务，可能还得往下细化。比如数字也分很多种，普通数字，手机号码，座机号码，热线号码、银行卡号，QQ号，微信号，金钱，距离等等，并且很多任务中，这些还可以单独作为一维特征。还得考虑中文数字阿拉伯数字等
  + 中文将字转换成拼音，许多恶意文本中会用同音字替代
  + 如果是英文的，那可能还得做词干提取、形态还原等，比如fucking,fucked -> fuck

+ 分词

+ 停用词

+ 数据平衡
  + 数据量不平衡
  + 数据多样性不平衡

+ 调整比例权重
  + https://www.jiqizhixin.com/articles/021704?from=synced&keyword=%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%8D%E5%B9%B3%E8%A1%A1

  

## 不平衡数据处理

+ https://www.zhihu.com/question/59236897

+ 不平衡类别的评估

  + AUC_ROC
  + mean Average Precesion （mAP）
    + 指的是在不同召回下的最大精确度的平均值
  + Precision@Rank k
    + 假设共有*n*个点，假设其中*k*个点是少数样本时的Precision。这个评估方法在推荐系统中也常常会用

+ 传统处理方法

  + UnderSampling
  + OverSampling
    + SMOTE
  + He, H. and Garcia, E.A., 2008. Learning from imbalanced data. IEEE Transactions on Knowledge & Data Engineering, (9), pp.1263-1284.
  + Roy, A., Cruz, R.M., Sabourin, R. and Cavalcanti, G.D., 2018. A study on combining dynamic selection and data preprocessing for imbalance learning. Neurocomputing, 286, pp.179-192.

+ 其他方法

  + 有监督的集成学习
    + 使用采样的方法建立K个平衡的训练集，每个训练集单独训练一个分类器，对K个分类器取平均
    + 一般在这种情况下，每个平衡的训练集上都需要使用比较简单的分类器（why？？？）
    + 但是效果不稳定
  + 无监督的异常检测
    + 从数据中找到异常值，比如找到spam
    + 前提假设是，spam 与正常的文章有很大不同，比如欧式空间的距离很大
    + 优势，不需要标注数据
    + https://www.zhihu.com/question/280696035/answer/417091151
    + https://zhuanlan.zhihu.com/p/37132428
  + [半监督集成学习](https://www.zhihu.com/question/59236897)
    + 未理解
    + 结合 有监督集成学习 和 无监督异常检测 的思路
    + 简单而言，你可以现在原始数据集上使用多个无监督异常方法来抽取数据的表示，并和原始的数据结合作为新的特征空间。在新的特征空间上使用集成树模型，比如xgboost，来进行监督学习。无监督异常检测的目的是提高原始数据的表达，监督集成树的目的是降低数据不平衡对于最终预测结果的影响。这个方法还可以和我上面提到的主动学习结合起来，进一步提升系统的性能。当然，这个方法最大的问题是运算开销比较大，需要进行深度优化。
  + [高维数据的半监督异常检测](Pang, G., Cao, L., Chen, L. and Liu, H., 2018. Learning Representations of Ultrahigh-dimensional Data for Random Distance-based Outlier Detection. arXiv preprint arXiv:1806.04808.)
    + 考虑到文本文件在转化后往往维度很高，可以尝试一下最近的一篇KDD文章
    + 主要是找到高维数据在低维空间上的表示，以帮助基于距离的异常检测方法

  + 结论
    + 直接在数据上尝试有监督的集成学习（方法1）
    + 直接在数据上使用多种无监督学习，观察哪一类算法的效果更好（方法2）
    + 结合以上两点(方法3)
    + 如果以上方法都不管用，尝试方法4
    + 使用方法1, 3，4时，可以加入主动学习
    + 如果以上方法不奏效， 则优先标注数据，提高数据质量
    + 监督学习>半监督>无监督

## 改变loss

+ weighted cross entropy

+ Focal loss
  Weighted loss function，一个处理非平衡数据常用的方法就是设置损失函数的权重，使得少数类判别错误的损失大于多数类判别错误的损失。在python的sk-learn中我们可以使用class_weight参数来设置权重，提高少数类权重，例如设置为多数类的10倍。

  RBG和Kaiming给出的相当牛逼的方法，这里不做详细介绍。 

  详情见链接：http://blog.csdn.net/u014380165/article/details/77019084

+ 特殊的过采样

  + <https://blog.csdn.net/u014535908/article/details/79035653>
  

## UDA 无监督数据扩充
+ https://github.com/google-research/uda
+ https://github.com/google-research/bert
+ Unsupervised Data Augmentation for Consistency Training [pdf](https://arxiv.org/abs/1904.12848)



## API

+ SMOTE
+ imblance learn
+ https://www.dataivy.cn/blog/3-4-%E8%A7%A3%E5%86%B3%E6%A0%B7%E6%9C%AC%E7%B1%BB%E5%88%AB%E5%88%86%E5%B8%83%E4%B8%8D%E5%9D%87%E8%A1%A1%E7%9A%84%E9%97%AE%E9%A2%98/

## Keypoint

+ 一般进行验证的时候都是验证集的acc或loss， 类别不平衡会导致验证集的这些指标不合理
  + 考虑极端情况：1000个训练样本中，正类样本999个，负类样本1个。训练过程中在某次迭代结束后，模型把所有的样本都分为正类，虽然分错了这个负类，但是所带来的损失实在微不足道，accuracy已经是99.9%，于是满足停机条件或者达到最大迭代次数之后自然没必要再优化下去，ok，到此为止，训练结束！于是这个模型……
+ 所以可以考虑换 验证集的指标，或者平衡数据
+ 指标有
  + 不平衡类别的评估
    - AUC_ROC
    - mean Average Precesion （mAP）
      - 指的是在不同召回下的最大精确度的平均值
    - Precision@Rank k
      - 假设共有*n*个点，假设其中*k*个点是少数样本时的Precision。这个评估方法在推荐系统中也常常会用
+ 平衡数据
  + 传统的Over sampling 和 Under sampling
  + 新的方法
    + SMOTE（Over sampling）
    + 有监督/无监督/半监督/高维降维