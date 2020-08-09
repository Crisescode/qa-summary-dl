## qa-summary

### 介绍
这个是百度AI Studio问答摘要与推理比赛实现算法，要求使用汽车大师提供的11万条技师与用户的多轮对话与诊断建议报告数据建立模型，基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，考验模型的归纳总结与推断能力。

* `./models/seq2seq.py`: 这个是纯用seq2seq的实现方式，没有加入attention注意力机制。效果最差，通过`ROUNAGE_L`评分才15.6左右；
* `./models/seq2seq_attention.py`: 这个是在seq2seq基础上加入 attention 注意力机制的实现方式。这个模型可以作为一个baseline结果，评分大概在28.5左右，以后模型可以基于这个结果不断优化；

### requirements
* python 3.0+
* tensorflow 2.0+
* numpy
* pandas 

### 使用方法


### 效果对比
