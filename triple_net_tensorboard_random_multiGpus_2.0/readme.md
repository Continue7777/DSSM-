阶段总结：

根据目前的情况，已经大致知道如何判断网络是否过拟合，以及已经摸清了tensorflow相关环境，并搭建基础框架。下一步的目标，制定标准化的测试方案以及pipeline的搭建，一次搭建，并行测试。主要包含如下几个部分：

- 输入层配置(over)
  - one-hot 
  - one-hot+ngram：ngram词典构建需要做成流程   ngram_flag（放在dataset），注意是否ngram返回长度不同
  - 词频：word_frequence_flag （放在dataset）  这个
- model层配置
  - 是否并行化也作为选项 √ 自动选择
  - 层结构支持1pos-n neg : 修改triplet_loss √
  - 模型要可一键并行化:现在的修改不用太多 √
  - loss/pred的参数最好不要全局，这样统一修改不方便。√
  - 需要支持任意模型：模型修改在get_loss函数 - 需要抽象一个model,pred loss evaluate都调用这个 √
- loss(over)
  - triplet loss : 切分distance函数和loss函数 √
  - softmax loss √
- 优化
  - 不同的优化函数选用：提供优化参数选项 √
- 二次训练
  - 支持many hard negative：先支持单个，一定时间后，只拿negtive去训练，这个先看效果后续再改进。
- 测评(这个稍微麻烦一点)
  - topn badcase输出：默认false
  - train_acc test_acc:   默认取best
  - knn_test_acc： 这个预先需要构建数据集。
  - 低置信度提取，做AUC图：
- 配置运行log
  - 搭建完成后，每一次的运行结果需要输出到一个文件
  - 内容
    - 网络结构：model()  model_fc()
    - 输出设置：word_frequent_flag、ngram_flag
    - hard_negative：begin_step
    - 网络参数：unit个数、是否bn、fc层数
    - 优化函数：str选取
    - 参数：bs、lr、step、print_step
    - 模型名，summary名字，路径
- 阶段性目标：
  - top1提高
  - 认识many hard negtive 的作用
  - 验证修改triplet - nplet的思路
  - 尝试不同优化器的作用
  - 尝试不同特征混入后的模型能力
  - 尝试引入cnn到结构中
  - 提升编码能力
- 论文阅读
  - 有关参数分布和神经网络的知识
  - triplet loss论文 ranking loss
  - text cnn
- 时间规划
  - 4.13日前完成上述所有编码+测试+结论总结



修改log:

data_input_fast_random_v2.py

- n_gram、word_freq的输入配置，接口不变。

utils_multi_gpu_v2.py

- 提出cos_distance_layer() 、 softmax_loss(),简化不同输入loss代码。
- triplet_loss 没有prob输出。
- model() 、 model_loss() 、model_pred_label() 简化run_with_session

参数设置注意事项：

- eular-triplet-loss   cos-softmax 
- 多negative 输入时，需要注意此时triplet-loss不适用

可视化修改：

- 只显示第一块卡上的weight图
- 添加训练集的test