NLP 仓库

### VSM：

复现论文Entropy-based Term Weighting Schemes for Text Categorization in VSM的文本分类实验

使用语料库 Reuters R8

使用模型 ： KNN、SVM

实现经典特征权重计算方法：

1. tf-idf
2. tf-chi
3. tf-rf
4. tf-ig
5. tf-eccd
6. iqf_qf_icf

实现论文提出的两种新的权重计算方法：

1. tf-dc
2. tf-bdc

整个实现过程详见[这篇文章](http://zedom1.top/2018/06/01/VSM/)

### Du Reading Comprehension

百度机器阅读理解 神经网络和xgboost调参部分代码

神经网络包含[M-LSTM](https://arxiv.org/abs/1608.07905)和[BIDAF](https://arxiv.org/abs/1611.01603)两个官方给出的baseline

在baseline基础上改成从每篇文章抽取一个答案，增加[r-net](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf)的核心层：self-matching

改动基于baseline的tensorflow版本

比赛总结详见[这篇文章](http://zedom1.top/2018/05/22/RC_DuReader/)