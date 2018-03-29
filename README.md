# DSSM
Dssm for tensorflow text classification

tensorflow 1.3.0
python 2.7

目标：使用dssm对分本进行分类

网络架构：one-hot(字) - fc1 - bn - fc2 - bn - loss(-log(prob))

toy下面是适用于小数据集的代码（之前没有优化，大数据时很多函数执行慢）

bigdata下面针对大数据的处理做了相应的优化。



bug：分析

+ loss不下降原因：
  + loss从一开始就不下降，表明数据流中可能出现bug，这里包含字典构建，词向量输出等Bug。如果确保数据和网络无误的情况下，loss不下降可能是loss本身的表示有问题，这里cos很小，没有*gamma=20时，loss不明显，加上学习率不够，导致整个loss基本无变化。
+ loss到最后下降不到位：
  + 进入局部最优，这表明优化step上是否可以有所改进。
  + 该词嵌入和网络的表达能力已经到了上线，修改网络或者增强词表达能力。比如把One-hot 改为word2vec.