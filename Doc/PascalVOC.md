# Pascal VOC

标签（空格分隔）： notes

---

## 文件夹格式（VOC数据集下有三个文件夹）
### 1.Annotations文件夹：存放标注，文件格式为.xml；
### 2.JPEGImages文件夹：存放数据图片，必须为jpg/jpeg格式，命名一般为6位数字编码：000001.jpg等；
### 3.ImageSets文件夹：其下有一个Main文件夹，其下一般有4个.txt文件，存放图像号码：
> * train.txt:训练图像号码；
> * val.txt:验证图像号码（有答案可对比）；
> * trainval.txt:以上两部分总和；
> * test.txt:测试图像号码；

## 本次数据较少，只获取train和val两个数据集，**数据抽取代码**和**数据清洗代码**由老师自己编写。




