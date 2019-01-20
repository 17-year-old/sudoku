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
        Mat src = Imgcodecs.imread(img);
        if (src.empty()) {
            System.out.println("加载图片出错!");
            return null;
        }

        Mat[][] matData = getMatData(src);
        int[][] data = new int[9][9];
        for (int row = 0; row < 9; row++) {
            for (int col = 0; col < 9; col++) {
                if (matData[row][col] == null) {
                    continue;
                }
                data[row][col] = findNearest(matData[row][col]);
            }
        }
        return data;
    }

    //处理图片
    //先查找定位出表格位置
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
        //Imgcodecs.imwrite("D:\\work\\sudoku\\src\\main\\resources\\thresh.png", thresh);

        //找轮廓
        //Imgproc.RETR_TREE表示找到所有的轮廓，不只是顶级轮廓
        //hierarchy表示各个轮廓之间的层级关系
        //hierarchy.get(0, i)[0]表示第i个轮廓的前一个轮廓
        //hierarchy.get(0, i)[1]表示第i个轮廓的后一个轮廓
        //hierarchy.get(0, i)[2]表示第i个轮廓的子轮廓
        //hierarchy.get(0, i)[3]表示第i个轮廓的父轮廓
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(thresh, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE, new Point(0, 0));
        //循环所有找到的轮廓-点
        List<MatOfPoint> contours_poly = contours;

        int tableArea = thresh.width() * thresh.height();
        int datacell = -1;
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint point = contours.get(i);
            MatOfPoint contours_poly_point = contours_poly.get(i);
            double area = Imgproc.contourArea(contours.get(i));
            //如果小于某个值就忽略，代表是杂线不是表格
            if (area < 100) {
                continue;
            }
            if (((int) hierarchy.get(0, i)[3] == -1) && (hierarchy.get(0, i)[2] > 0)) {
                if (area > tableArea / 4) {
                    datacell = (int) hierarchy.get(0, i)[2];//这里取的是表格的子轮廓
                    break;
                }
            }
        }

        Mat[][] mat_data = new Mat[9][9];
        int i = 80, j = 0;
        while (datacell > 0) {
            int row = i / 9;
            int col = i % 9;
            int child = (int) hierarchy.get(0, datacell)[2];
            if (child > 0) {//某个单元格的子轮廓为空，说明这个单元个为空
                MatOfPoint point = contours.get(child);
                MatOfPoint contours_poly_point = contours_poly.get(child);
                /*
                 * approxPolyDP 函数用来逼近区域成为一个形状，true值表示产生的区域为闭合区域。比如一个带点幅度的曲线，变成折线
                 *
                 * MatOfPoint2f curve：像素点的数组数据。
                 * MatOfPoint2f approxCurve：输出像素点转换后数组数据。
                 * double epsilon：判断点到相对应的line segment 的距离的阈值。（距离大于此阈值则舍弃，小于此阈值则保留，epsilon越小，折线的形状越接近曲线。）
                 * bool closed：曲线是否闭合的标志位。
                 */
                Imgproc.approxPolyDP(new MatOfPoint2f(point.toArray()), new MatOfPoint2f(contours_poly_point.toArray()), 3, true);
                //为将这片区域转化为矩形，此矩形包含输入的形状
                Rect boundRect = Imgproc.boundingRect(contours_poly_point);
                temp = thresh.submat(boundRect);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                Imgcodecs.imwrite("D:\\work\\sudoku\\src\\main\\resources\\cell" + j++ + ".png", temp);
                mat_data[row][col] = temp;
            }
            //忽略面积小的单元格
            double area = Imgproc.contourArea(contours.get(datacell));
            //转到下一个轮廓, 轮廓的顺序
            datacell = (int) hierarchy.get(0, datacell)[0];
            if (area > 100) {
                i--;
            }
        }
        return mat_data;
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

    //生成训练样本
    private static Mat[][] makeSamples() {
        samples = new Mat[10][48];
        for (int i = 0; i <= 9; i++) {
            for (int j = 0; j <= 7; j++) { //不同的字体
                //
                Mat temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j, 1.0, new Scalar(255), 2, Imgproc.LINE_4, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6] = temp;

                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j, 1.0, new Scalar(255), 2, Imgproc.LINE_8, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6 + 1] = temp;

                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j, 1.0, new Scalar(255), 2, Imgproc.LINE_AA, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6 + 2] = temp;

                //对应的斜体
                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j + 16, 1.0, new Scalar(255), 2, Imgproc.LINE_4, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6 + 3] = temp;

                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j + 16, 1.0, new Scalar(255), 2, Imgproc.LINE_8, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6 + 4] = temp;

                temp = Mat.zeros(50, 50, 0);
                Imgproc.putText(temp, String.valueOf(i), new Point(5, 25), j + 16, 1.0, new Scalar(255), 2, Imgproc.LINE_AA, false);
                temp = cut(temp);
                Imgproc.resize(temp, temp, new Size(32, 32), 0, 0, Imgproc.INTER_AREA);
                samples[i][j * 6 + 5] = temp;
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
    private static int findNearest(Mat feature) {
        Mat res = new Mat();
        Mat temp = feature;
        feature.convertTo(temp, CvType.CV_32F);
        temp = temp.reshape(1, 1);
        float p = knn.findNearest(temp, 1, res);
        return (int) p;
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

}
