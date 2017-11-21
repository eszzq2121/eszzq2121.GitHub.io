---
layout: post
title:  "Classification(1)：kNN"
date:   2017-11-21
excerpt: "kNN application"
image: "/images/kNN.png"
---

## Introduction

这段时间在CSDN上报了一个数据挖掘的课程，开这个网站的一个很大的目地也是想要找个地方督促自己整理学习的内容，如果我能坚持下来的话，博客部分可能以这些内容为主，改网页的时候顺便学习一下html的写法。
我报的这个课程，因为时间的关系，对于算法类主要以了解原理为主，没有详细介绍，也没有对算法进行扩展，这段时间闲下来就课后一个一个学，从kNN开始
							


## 1. 分类算法的应用场景

分类算法是数据挖掘技术中常用的监督式学习工具，主要基于包含特征和具体类别信息的历史样本数据，建立特征-类别之间的规律，并对新个体进行预测。典型的问题包括：

- 鸢尾花分类
- 验证码识别
- 客户分类
- 销量预测
- 录取概率

我们以鸢尾花分类为问题背景，简单介绍分类问题的一般模型，并在此基础上介绍一些常用的分类算法：

现在已知一个随机样本中三种鸢尾花（Setosa  Versicolor  Virginica）对应的花萼长度宽度、花瓣长度宽度这四个特征变量，

我们的目的是希望能够根据历史的样本信息判断出一个新的鸢尾花样本对应的种类。显然，这是一个多分类问题，而kNN是解决这类问题的一个经典的算法

## 2. kNN算法
### （1）算法原理
##### 1.计算新个体到旧数据之间的距离
距离的定义：欧氏距离
对于文本分类来说，使用余弦(cosine)来计算相似度比欧式距离更合适
##### 2.统计出距离最短的前K个商品 
K定义的原则：3 5 10…… 根据样本数量而定
经验规则：k一般低于训练样本数的平方根
##### 3.统计距离最短的前K个商品中哪一个类别最多
投票决定：少数服从多数
加权投票法：根据距离的远近，距离越近则权重越大（权重为距离平方的倒数）
##### 4.将新商品归为类别最多的类别
							  
#### （2）kNN算法                             
基于欧氏距离、投票决定的规则,以下为原生代码的实现
```							
from numpy import *
import operator
def knn(k,testdata,traindata,labels) #traindata和labels 分别存储特征和类别
   #testdata : 1d array   [0,0,1,0,0,1,1,.....]
   #traindata :2d array [[1,0,0,.....][0,0,1,1,......][][]......]
   #labels :   list  , match to traindata
   traindatasize=traindata.shape[0]  
   # "shape" define the feature of the traindata 定义数组的数量，测试集有多少个记录
   dif=tile(testdata,(traindatasize,1))-traindata  
   #tile()将一维的测试数据转为与训练数据一样的行列格式，替代for循环
   sqdif=dif**2
   sumsqdif=sqdif.sum(axis=1) 
   #axis=1 表明横向相加,此时 sumsqdif里面的数组是一维的
   distance=sumsqdif**0.5
   sortdistance=distance.argsort() 
   #由近到远排序的结果   argsort函数返回的是数组值从小到大的索引值
   count={}  
   #定义一个字典
   for i in range(0,k): 
   #k是自己定义的
       vote=labels[sortdistance[i]] 
	   #排序靠前的对应的类
	   count[vote]=count.get(vote,0)+1  
	   #获取labels对应的字典值，如果没有定义赋0值，每循环到一次加1   dict.get()的用法
  sortcount=sorted(count.items(),key=operator.itemgetter(1),reverse=True) 
  #key指定的是sorted函数中的比较对象，itemgetter获取字典中的属性，1位置即前面累计的次数，reverse=True指定降序排列
  return sortcount[0][0] 
  #新字典中第一个记录的label名称即为KNN算法的结果 
```  
  
在sklearn包中提供了现成的kNN算法的函数，使用起来非常方便
```  
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x,y)
y2=model.predict(x2)
```                      
kNN算法属于懒惰算法，不适用于大规模的数据量，用kd数构造的kNN方法能有效降低算法的运算次数，在sklearn包中有相应的参数设置

## 补充学习资料：
[Sorted函数及Operator模块](http://blog.csdn.net/myarrow/article/details/51200167)
[KNN的理解](https://www.joinquant.com/post/2227?f=zh)
[KD树的构造以及KD树上的 kNN 算法](https://www.joinquant.com/post/2843)
[Sklearn中KNN算法的应用进阶](https://zhuanlan.zhihu.com/p/23191325)