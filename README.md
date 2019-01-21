# sudoku
解数独的程序,可以从图片中读取题目并给出答案
使用opencv,通过findContours寻找轮廓然后分解成小的单元格
再使用KNN判断,生成sudoku数组,KNN样本是通过Imgproc.putText生成的
目前还识别不了复杂的表结构,如sudoku7.png各个单元格未嵌套在同一个大的表格中,暂时不能处理
只测试了电脑截图,未测试手机照片,所以也没考虑对倾斜的图片的处理  

