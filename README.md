# Corpus

## Dataset for training
+ https://keras.io/zh/datasets/

# Preprocessing

+ A First-level folder for Preprocessing, contains Data Augmentation, Data Clean, Data Translate, Data Visualization

  | Data Annotator  | Data Augmentation | Data Clean | Data Translate | Data Visualizaiton |
  | --------------- | ----------------- | ---------- | -------------- | ------------------ |
  | BRAT            | Over Sampling     | punction   |                |                    |
  | Active Learning | Under Sampling    | stop-words |                |                    |
  |                 | Data-Generation   | vocablary  |                |                    |
  |                 |                   |            |                |                    |

## Reference
### 小样本不平衡样本
- 训练一个有效的神经网络，通常需要大量的样本以及均衡的样本比例，但实际情况中，我们容易的获得的数据往往是小样本以及类别不平衡的，比如银行交易中的fraud detection和医学图像中的数据。前者绝大部分是正常的交易，只有少量的是fraudulent transactions；后者大部分人群中都是健康的，仅有少数是患病的。因此，如何在这种情况下训练出一个好的神经网络，是一个重要的问题。
  本文主要汇总训练神经网络中解决这两个问题的方法。
+ Training Neural Networks with Very Little Data - A Draft -arxiv,2017.08
  + “Training Neural Networks with Very Little Data”学习笔记

### View of Corpus too big
+ https://www.leiphone.com/news/201705/sghfB2wSub6W01Jy.html

### 使用无监督和半监督来减少标注
+ https://blog.csdn.net/lujiandong1/article/details/52596654

### 结构化数据标记
+ 一般采用json-ld 格式
+ 对非结构化数据进行组织时, 一般使用schema.org 定义的类型和属性作为标记(比如json-ld),且此标记要公开

### Schema.org 
+ https://www.ibm.com/developerworks/cn/web/wa-schemaorg1/index.html
	+ 最重要的是，这样做可以使您的页面更容易访问，更容易通过搜索引擎、AI 助手和相关 Web 应用程序找到。您不需要学习任何新的开发系统或工具来使用标记，而且在几小时内就可以快速上手

### Data Format
+ iconv -f gb2312 -t utf-8 aaa.asp -o bbb.asp
+ iconv -f utf-16 -t utf-8 train.txt -o train.txt
