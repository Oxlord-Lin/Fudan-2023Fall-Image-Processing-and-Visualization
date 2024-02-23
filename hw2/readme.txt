report.pdf是本次作业的实验报告，内有Python代码以及实验结果

codes文件夹中，包含了本次作业的Python源代码，辅助文件，以及相应的测试图与实验结果图。具体如下：

- hw1_exercise2_2是来自第一次作业的文件，用于导入高效生成局部灰度直方图的函数local-hist，【可以忽略】

- exercise_2.py文件是用于解决第二道题局部二值化的python源代码

- 肝脏.png 是exercise_2.py文件的测试图

- exercise_3.py文件是用于解决第三道题线性插值提高图片分辨率的python源代码

- 86版西游记模糊剧照.png 是用于exercise_3.py文件的测试图，该图的分辨率很小

codes文件夹中的以下文件【可以忽略】：

- pixels after thresholding文件夹存储的是能够被numpy库读取的矩阵文件，由exercise_2.py产生的二值化后的图片灰度值都存储在这个文件夹下，可以忽略这个文件夹

- some local histograms则存储了肝脏图片的对应于不同patch size的局部灰度直方图，可以在exercise_2.py中直接导入，可以节省一些运行时间，具体导入方式请看exercise_2.py的注释说明。如果不想直接导入已生成的局部灰度直方图，也可以忽略这个文件夹

- 剩余的其他图片都是本次实验的结果图片，均已纳入报告中，可以忽略