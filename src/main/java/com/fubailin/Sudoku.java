package com.fubailin;

import java.util.ArrayList;
import java.util.List;

public class Sudoku {
    //用数组保存sudoku数据
    //对于元素sudoku[x][y]
    //x表示纵坐标, y表示横坐标, 值0表示空
    private static int[][] sudoku = new int[9][9];
    private static List<Object> out = new ArrayList<Object>();

    private static boolean check(int x, int y, int v) {
        //检查同一行是否有重复值
        for (int i = 0; i < 9; i++) {
            if (sudoku[x][i] == v) {
                return false;
            }
        }
        //检查同一列是否有重复值
        for (int i = 0; i < 9; i++) {
            if (sudoku[i][y] == v) {
                return false;
            }
        }

        //检查同一个小方阵的是否有重复值
        //先计算小方阵坐标
        //0-2, 3-5, 6-8属于同一个小方阵
        int beginX = x / 3 * 3;
        int beginY = y / 3 * 3;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                if (sudoku[beginX + i][beginY + j] == v) {
                    return false;
                }
            }
        }
        return true;
    }

    private static void solveNext(int x, int y) {
        int nexty = (y + 1) % 9;
        int nextx = x;
        if (y == 8) {
            nextx = x + 1;
        }
        solve(nextx, nexty);
    }

    private static void solve(int x, int y) {
        //sudoku已经解开
        if (x == 9 && y == 0) {
            int[][] result = new int[9][9];
            for (int i = 0; i < 9; i++) {
                for (int j = 0; j < 9; j++) {
                    result[i][j] = sudoku[i][j];
                }
            }
            out.add(result);
            return;
        }
        //这个单元格已经有值了,直接返回
        if (sudoku[x][y] != 0) {
            solveNext(x, y);
            return;
        }

        //解x,y这个单元格
        //从可用的备选数中依次进行实验
        for (int i = 1; i < 10; i++) {
            if (check(x, y, i)) {
                //这个解不冲突,解下一个单元格
                sudoku[x][y] = i;
                solveNext(x, y);
                //上一步会在成功解出或者解失败后返回
                //将本单元格设置为空，进行下一个尝试，这样可以找到多个解
                sudoku[x][y] = 0;
            }
        }
    }

    private static int count() {
        int i = 0;
        for (int x = 0; x < 9; x++) {
            for (int y = 0; y < 9; y++) {
                if(sudoku[x][y] >0 ) {
                    i++;
                }
            }
        }
        return i;
    }

    public static List<Object> solve() {
        solve(0, 0);
        return out;
    }

    public static void setSudokuData(int[][] data) {
        assert (data.length == 9);
        assert (data[0].length == 9);
        sudoku = data;
    }

    public static List<Object> solve(int[][] data) {
        setSudokuData(data);
        if(count() <17){
            System.out.println("题目提供的数字太少！");
            return out;
        }
        solve();
        return out;
    }
}
