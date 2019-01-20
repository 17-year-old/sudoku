package com.fubailin;

import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.ml.KNearest;
import org.opencv.ml.Ml;
import org.opencv.utils.Converters;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

public class ScanSudoku {
    private static Mat[][] samples;
    private static KNearest knn;
    private static String path = "D:\\work\\sudoku\\src\\main\\resources\\";

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        samples = makeSamples();
        knn = KNearest.create();
        train();
    }

    //扫描图片，返回sudoku的数组
    public static int[][] scanSudoku(String img) {
        Mat src = Imgcodecs.imread(path + "sudoku.png");
        if (src.empty()) {
            System.out.println("加载图片出错!");
            return null;
        }

        Mat[][] matData = getMatData(src);
        int[][] data = new int[9][9];
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                if (getPiexSum(matData[row][col]) == 0) {
                    continue;
                }
                data[row][col] = findNearest(matData[row][col]);
            }
        }
        return data;
    }

    //处理图片
    //将图片分割成9*9的Mat数组
    private static Mat[][] getMatData(Mat src) {
        Mat gray = new Mat();
        //灰度处理
        Imgproc.cvtColor(src, gray, Imgproc.COLOR_BGR2GRAY);
        //二值化
        Mat thresh = gray.clone();
        Mat temp = new Mat();
        Core.bitwise_not(gray, temp);
        Imgproc.adaptiveThreshold(temp, thresh, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 15, -2);

        //克隆一个 Mat，用于提取水平线
        Mat horizontal = thresh.clone();
        //克隆一个 Mat，用于提取垂直线
        Mat vertical = thresh.clone();
        /*
         * 求水平线
         * 1. 根据页面的列数（可以理解为宽度），将页面化成若干的扫描区域
         * 2. 根据扫描区域的宽度，创建一根水平线
         * 3. 通过腐蚀、膨胀，将满足条件的区域，用水平线勾画出来
         *
         * scale 越大，识别的线越多，因为，越大，页面划定的区域越小，在腐蚀后，多行文字会形成一个块，那么就会有一条线
         * 在识别表格时，我们可以理解线是从页面左边 到 页面右边的，那么划定的区域越小，满足的条件越少，线条也更准确
         */
        int scale = 10;

        // 为了获取横向的表格线，设置腐蚀和膨胀的操作区域为一个比较大的横向直条
        int horizontalsize = horizontal.cols() / scale;
        Mat horizontalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(horizontalsize, 1));
        // 先腐蚀再膨胀 new Point(-1, -1) 以中心原点开始
        // iterations 最后一个参数，迭代次数，越多，线越多。在页面清晰的情况下1次即可。
        Imgproc.erode(horizontal, horizontal, horizontalStructure, new Point(-1, -1), 1);
        Imgproc.dilate(horizontal, horizontal, horizontalStructure, new Point(-1, -1), 1);

        /// 求垂直线
        int verticalsize = vertical.rows() / scale;
        Mat verticalStructure = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, verticalsize));
        Imgproc.erode(vertical, vertical, verticalStructure, new Point(-1, -1), 1);
        Imgproc.dilate(vertical, vertical, verticalStructure, new Point(-1, -1), 1);

        //合并线条,将垂直线，水平线合并为一张图
        Mat mask = new Mat();
        Core.add(horizontal, vertical, mask);

        //通过 bitwise_and 定位横线、垂直线交汇的点
        Mat joints = new Mat();
        Core.bitwise_and(horizontal, vertical, joints);

        //找轮廓
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(mask, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));

        List<MatOfPoint> contours_poly = contours;
        Rect[] boundRect = new Rect[contours.size()];
        LinkedList<Mat> tables = new LinkedList<Mat>();
        //循环所有找到的轮廓-点
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint point = contours.get(i);
            MatOfPoint contours_poly_point = contours_poly.get(i);
            /*
             * 获取区域的面积
             * 第一个参数，InputArray contour：输入的点，一般是图像的轮廓点
             * 第二个参数，bool oriented = false:表示某一个方向上轮廓的的面积值，顺时针或者逆时针，一般选择默认false
             */
            double area = Imgproc.contourArea(contours.get(i));
            //如果小于某个值就忽略，代表是杂线不是表格
            if (area < 100) {
                continue;
            }
            /*
             * approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。比如一个带点幅度的曲线，变成折线
             *
             * MatOfPoint2f curve：像素点的数组数据。
             * MatOfPoint2f approxCurve：输出像素点转换后数组数据。
             * double epsilon：判断点到相对应的line segment 的距离的阈值。（距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越“接近”曲线。）
             * bool closed：曲线是否闭合的标志位。
             */
            Imgproc.approxPolyDP(new MatOfPoint2f(point.toArray()), new MatOfPoint2f(contours_poly_point.toArray()), 3, true);
            //为将这片区域转化为矩形，此矩形包含输入的形状
            boundRect[i] = Imgproc.boundingRect(contours_poly.get(i));
            // 找到交汇处的的表区域对象
            Mat table_image = joints.submat(boundRect[i]);

            List<MatOfPoint> table_contours = new ArrayList<MatOfPoint>();
            Mat joint_mat = new Mat();
            Imgproc.findContours(table_image, table_contours, joint_mat, Imgproc.RETR_CCOMP, Imgproc.CHAIN_APPROX_SIMPLE);
            //从表格的特性看，如果这片区域的点数小于4，那就代表没有一个完整的表格，忽略掉
            if (table_contours.size() < 4)
                continue;
            //保存图片
            tables.addFirst(thresh.submat(boundRect[i]).clone());
            mask = mask.submat(boundRect[i]).clone();
        }

        //有多个table的话,选择最大的table
        Mat sudoku = null;
        for (int i = 0; i < tables.size(); i++) {
            if (sudoku == null) {
                sudoku = tables.get(i);
            } else {
                if (tables.get(i).size().area() > sudoku.size().area()) {
                    sudoku = tables.get(i);
                }
            }
        }

        Mat[][] mat_data = new Mat[9][9];
        int colwidth = sudoku.width() / 9;
        int rowheight = sudoku.height() / 9;
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                Rect r = new Rect(colwidth * col, rowheight * row, colwidth, rowheight);
                Mat t = sudoku.submat(r);
                t = removeBlackEdge(t);
                t = cut(t);
                Imgproc.resize(t, t, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                mat_data[row][col] = t;
            }
        }
        return mat_data;
    }

    //删除黑边,这里黑边是从表格线中截取出来的
    //这里处理方式很简单
    //删除左边和顶部8个像素宽度
    //删除右边和下边2个像素宽度
    public static Mat removeBlackEdge(Mat t) {
        int left = 0, top = 0, width = 0, height = 0;
        left = 8;
        top = 8;
        width = t.height() - 10;
        height = t.height() - 10;

        Rect r = new Rect(left, top, width, height);
        return t.submat(r);
    }

    //去除周围空白,只保留包括了文字的矩形部分
    public static Mat cut(Mat t) {
        int left = 0, right = t.width() - 1, top = 0, bottom = t.height() - 1;

        for (int col = 0; col < t.width(); col++) {
            int colsum = 0;
            for (int row = 0; row < t.height(); row++) {
                colsum = (int) (colsum + t.get(row, col)[0]);
            }
            if (colsum > 0) {
                left = col;
                break;
            }
        }

        for (int col = t.width() - 1; col >= 0; col--) {
            int colsum = 0;
            for (int row = 0; row < t.height(); row++) {
                colsum = (int) (colsum + t.get(row, col)[0]);
            }
            if (colsum > 0) {
                right = col;
                break;
            }
        }

        for (int row = 0; row < t.height(); row++) {
            int rowsum = 0;
            for (int col = 0; col < t.width(); col++) {
                rowsum = (int) (rowsum + t.get(row, col)[0]);
            }
            if (rowsum > 0) {
                top = row;
                break;
            }
        }

        for (int row = t.height() - 1; row >= 0; row--) {
            int rowsum = 0;
            for (int col = 0; col < t.height(); col++) {
                rowsum = (int) (rowsum + t.get(row, col)[0]);
            }
            if (rowsum > 0) {
                bottom = row;
                break;
            }
        }

        Rect r = new Rect(left, top, right - left, bottom - top);
        return t.submat(r);
    }

    //打印出二进制的mat
    private static void printMat(Mat t) {
        System.out.println();
        for (int row = 0; row < t.height(); row++) {
            for (int col = 0; col < t.width(); col++) {
                if ((int) t.get(row, col)[0] == 0) {
                    System.out.print("  0 ");
                } else {
                    System.out.print((int) t.get(row, col)[0] + " ");
                }
            }
            System.out.println();
        }
    }

    //求图片的二进制和
    private static double getPiexSum(Mat t) {
        double sum = 0;
        for (int row = 0; row < t.height(); row++) {
            for (int col = 0; col < t.width(); col++) {
                sum = sum + t.get(row, col)[0];
            }
        }
        return sum;
    }

    //生成训练样本
    private static Mat[][] makeSamples() {
        samples = new Mat[10][16];
        for (int i = 0; i <= 9; i++) {
            for (int j = 0; j <= 7; j++) {
                //不同的字体
                Mat temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j, 1.0, new Scalar(255), 2, Imgproc.LINE_8, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 2] = temp;
                Imgcodecs.imwrite("D:\\work\\sudoku\\src\\main\\resources\\samples\\" + i + "_" + (j * 2) + ".png", temp);

                //对应的斜体
                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j + 16, 1.0, new Scalar(255), 2, Imgproc.LINE_8, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 2 + 1] = temp;
                Imgcodecs.imwrite("D:\\work\\sudoku\\src\\main\\resources\\samples\\" + i + "_" + (j * 2 + 1) + ".png", temp);
            }
        }
        return samples;
    }

    //使用样本进行训练
    private static void train() {
        Mat trainData = new Mat();
        List<Integer> trainLabs = new ArrayList<Integer>();
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < samples[i].length; j++) {
                samples[i][j].convertTo(samples[i][j], CvType.CV_32F);
                trainData.push_back(samples[i][j].reshape(1, 1));
                trainLabs.add(i);
            }
        }
        knn.train(trainData, Ml.ROW_SAMPLE, Converters.vector_int_to_Mat(trainLabs));
    }

    //对数据进行匹配
    public static int findNearest(Mat feature) {
        Mat res = new Mat();
        Mat temp = feature;
        feature.convertTo(temp, CvType.CV_32F);
        temp = temp.reshape(1, 1);
        float p = knn.findNearest(temp, 1, res);
        return (int) p;
    }
}
