package com.fubailin;

import com.fubailin.opencv.BinaryUtils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class ScanSudoku {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat src = Imgcodecs.imread("D:\\work\\sudoku\\src\\main\\resources\\sudoku.png", Imgcodecs.IMREAD_GRAYSCALE);
        BinaryUtils.binaryzation(src);
        Imgcodecs.imwrite("D:\\work\\sudoku\\src\\main\\resources\\sudoku1.png", src);
        System.out.println(src);
    }
}
