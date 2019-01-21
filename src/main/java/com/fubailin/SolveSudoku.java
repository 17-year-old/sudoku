package com.fubailin;

import java.util.List;

public class SolveSudoku {
    public static void main(String[] args) {
        int[][] data = ScanSudoku.scanSudoku("C:\\work\\src\\sudoku\\src\\main\\resources\\sudoku0.png");
        for (int x = 0; x < 9; x++) {
            for (int y = 0; y < 9; y++) {
                System.out.print(data[x][y]);
            }
            System.out.println("");
        }
        List<Object> retult = Sudoku.solve(data);
        if (retult.isEmpty()) {
            System.out.println("无解!");
            return;
        }
        System.out.println("共有" + retult.size() + "解!");
        for (int index = 0; index < retult.size(); index++) {
            System.out.println("解" + (index + 1) + "!");
            int[][] t = (int[][]) retult.get(index);
            for (int x = 0; x < 9; x++) {
                for (int y = 0; y < 9; y++) {
                    System.out.print(t[x][y]);
                }
                System.out.println("");
            }
        }
    }
}
