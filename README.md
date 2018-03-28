# DSSM
Dssm for tensorflow text classification

tensorflow 1.3.0
python 2.7

目标：使用dssm对分本进行分类

网络架构：one-hot(字) - fc1 - bn - fc2 - bn - loss(-log(prob))

toy下面是适用于小数据集的代码（之前没有优化，大数据时很多函数执行慢）

bigdata下面针对大数据的处理做了相应的优化。