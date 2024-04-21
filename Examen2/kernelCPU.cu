#include <iostream>
#include <vector>
#include <chrono>

// Function: isValid
// Description: Checks if a given number is valid to be placed in a particular cell of the Sudoku board.
// Parameters:
// - board: The Sudoku board represented as a 2D vector of characters.
// - row: The row index of the cell being checked.
// - col: The column index of the cell being checked.
// - num: The number to be checked for validity in the cell.
// Returns:
// - bool: True if the number is valid in the cell, false otherwise.
bool isValid(const std::vector<std::vector<char>>& board, int row, int col, char num) {
    // Check row
    for (int i = 0; i < 9; ++i) {
        if (board[row][i] == num) return false;
    }

    // Check column
    for (int i = 0; i < 9; ++i) {
        if (board[i][col] == num) return false;
    }

    // Check subgrid
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = startRow; i < startRow + 3; ++i) {
        for (int j = startCol; j < startCol + 3; ++j) {
            if (board[i][j] == num) return false;
        }
    }

    return true;
}

// Function: solveSudoku
// Description: Recursively solves the Sudoku puzzle using backtracking algorithm.
// Parameters:
// - board: The Sudoku board represented as a 2D vector of characters.
// Returns:
// - bool: True if Sudoku is solved successfully, false if no solution exists.
bool solveSudoku(std::vector<std::vector<char>>& board) {
    for (int row = 0; row < 9; ++row) {
        for (int col = 0; col < 9; ++col) {
            if (board[row][col] == '.') {
                for (char num = '1'; num <= '9'; ++num) {
                    if (isValid(board, row, col, num)) {
                        board[row][col] = num;
                        if (solveSudoku(board)) return true;
                        board[row][col] = '.'; // Undo the choice
                    }
                }
                return false; // No valid number found
            }
        }
    }
    return true; // Sudoku solved
}

// Function: printSudoku
// Description: Prints the Sudoku board to the console.
// Parameters:
// - board: The Sudoku board represented as a 2D vector of characters.
void printSudoku(const std::vector<std::vector<char>>& board) {
    for (const auto& row : board) {
        for (char cell : row) {
            std::cout << cell << " ";
        }
        std::cout << std::endl;
    }
}

// Function: main
// Description: Entry point of the program. Solves a predefined Sudoku puzzle and prints the solution along with elapsed time.
int main() {
    // Sudoku board
    const std::vector<std::vector<char>> board = {
        {'5','3','.','.','7','.','.','.','.'},
        {'6','.','.','1','9','5','.','.','.'},
        {'.','9','8','.','.','.','.','6','.'},
        {'8','.','.','.','6','.','.','.','3'},
        {'4','.','.','8','.','3','.','.','1'},
        {'7','.','.','.','2','.','.','.','6'},
        {'.','6','.','.','.','.','2','8','.'},
        {'.','.','.','4','1','9','.','.','5'},
        {'.','.','.','.','8','.','.','7','9'}
    };

    std::vector<std::vector<char>> solvedBoard = board;

    // Start tracking time
    auto start = std::chrono::steady_clock::now(); // Start time measurement

    if (solveSudoku(solvedBoard)) {
        printSudoku(solvedBoard);
        auto end = std::chrono::steady_clock::now(); // End time measurement
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Elapsed time for CPU solving: " << elapsed.count() << " seconds" << std::endl;
    } else {
        std::cout << "No solution exists for the given Sudoku board.\n";
    }

    return 0;
}
