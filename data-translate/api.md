### 有道

+ http://fanyi.youdao.com/openapi?path=data-mode
+ 每小时 1000次
+ 需要申请

### 百度

+ 



### google
+ https://pypi.org/project/googletrans/
  + 有待测试

+ goslate

  """

  import goslate

  """

+ Google.cloud translate

  + 收费 
    + 20 $ / 1M 字符
  + https://blog.csdn.net/u010856630/article/details/73810718

  """

  from google.cloud import translate
  translate_client = translate.Client()

  text = u'hello,world'
  target = 'ru'

  translation = translate_client.translate(text,target_language=target)

  print u'Text:{}'.format(text)
  print u'translation:{}'.format(translation['translatedText'])

  """

+ Translate-api （线上测试不好用）

  +  <https://github.com/yixianle/google-translate>
  + <http://translate.hotcn.top/>