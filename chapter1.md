# Chapter 1: Introduction

Keywords: AI, deep learning, machine learning, representation learning

## Description
作者从AI讲起, 逐步延伸到深度学习的概念, 定义了本书语境下的深度学习这个概念.

作者首先论述了AI的难点在于解决那些对于人类很容易但是难以用严格的数学语言定义或者描述的问题, 也就是由人们直观解决的问题.

这个难点的解决方法是让计算机从经验中自己学习知识, 并且让计算机从一个概念的层次结构中理解世界, 也就是说从简单概念之间的联系开始学习, 逐步向上进行理解和学习. 这样也就无需人类提供形式化的知识. 因为概念的构成是一个层次结构, 所以知识的形态必然是一个具有相当深度的网状或者图状结构, 所以我们把这种思路下的AI称为AI深度学习.

这时作者又论述了另一个AI的挑战, 那就是如何把非正式的知识输送给计算机, 也就是说不直接给计算机定义知识了, 要让计算机从经验中学习知识, 那经验又如何告诉给计算机呢, 作者这里举了知识库的例子来说明硬编码知识传递的无力, 从而引出了机器学习的思路, 机器学习的方法赋予AI自己学习知识的能力, 但是作者这里又说明了机器学习的局限, 机器学习的特征需要自己来手动选择, 十分低效.

解决机器学习特征选择问题的方法是引入了表征学习(representation learning), 表征学习不仅把特征映射带输出上来, 也映射到自己上, 也就是它可以自己学习特征. 作者在这里举了一个自编码器(autoencoder)的例子.

但是由于对数据有影响的因素太多太复杂, 所以表征学习也很难学习到比较高层次, 比较抽象的特征, 所以这是也就引入来深度学习. 深度学习的解决方法是把高层次的特征用其它简单的低层次的特征来表示. 作者在这里举了前馈深度网络或者叫多层感知机的例子.

然后作者从两个视角来解释了深度学习, 第一种角度是深度学习是一个有许多简单函数组合而成的较为复杂的函数, 把输入数据映射到输入数据上来, 另一种视角是把深度学习看做一个多步骤的计算机程序, 计算过程中保留了状态信息以供后面的程序进行参考, 从而做出调整.

作者随后论述了度量深度学习深度的两种方法: 计算图(the depth of the computational graph)的深度和概率模型图的深度(the depth of the probabilistic modeling graph). 但是判断多深才能叫做深并没有一个绝对的指标.

最后作者点出了本书的主题, 我直接引用作者原文:
> To summarize, deep learning, the subject of this book, is an approach to AI.
> Specially, it is a type of machine learning, a technique that allows computer systems to improve 
> with experience and data.

### 1.1 Who Should Read This Book?
作者这里列出了他心目中的两类目标读者, 一类是正在学习机器学习的大学生(本科或者研究生均算在内), 另一类是亟需机器学习背景知识的软件工程师.

本书分成了三个部分:
1. 第一部分是基本的数学知识和机器学习基础
2. 第二部分是较为成熟的深度学习知识
3. 第三部分是目前还不太成熟, 但是非常有前景的深度学习知识

### 1.2 Historical Trends in Deep Learning

#### 1.2.1 The Many Names and Changing Fortunes of Neural Networks
这一小节, 主要是介绍了深度学习一路的发展. 深度学习在不同阶段的名称不尽相同:
1. 1940s-1960s: **cybernetics**
2. 1980s-1990s: **connectionism**
3. 2006-now: **deep learning**
具体历史细节, 不在此赘述.

#### 1.2.2 Increasing Dataset Sizes
这一小节说明了一个很重要的问题, 为什么机器学习的模型和训练方法在几十年前就已经出现了, 但到最近才兴起, 作者提到了社会的数字化(Digitization)让我们拥有了更多的数据可供使用. 数据集中数据的增多减轻了泛化的负担.

#### 1.2.3 Increasing Model Sizes
这一小节作者提到了促使机器学习在最近兴起的另一个重要原因是计算资源的丰富, 随着硬件水平的提高, 我们能够训练更庞大的网络, 所以加上更大的数据集使得机器学习的算法焕发了新的活力.

#### 1.2.4 Increasing Accuracy, Complexity and Real-World Impact
这一小节作者描述了深度学习在多个领域取得的辉煌成果, 精确度在很多任务上取得了较大的突破. 具体的数据不在此赘述.





