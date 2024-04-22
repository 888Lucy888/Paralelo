#include <iostream>
#include <vector>
#include <chrono>

#define N 9
#define GPUErrorAssertion(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Device function to check if a number is valid in a particular cell
__device__ bool isValid(char* board, int row, int col, char num) {
    // Check row
    for (int i = 0; i < N; ++i) {
        if (board[row * N + i] == num) return false;
    }

    // Check column
    for (int i = 0; i < N; ++i) {
        if (board[i * N + col] == num) return false;
    }

    // Check subgrid
    int startRow = row - row % 3;
    int startCol = col - col % 3;
    for (int i = startRow; i < startRow + 3; ++i) {
        for (int j = startCol; j < startCol + 3; ++j) {
            if (board[i * N + j] == num) return false;
        }
    }

    return true;
}

// Device function to solve the Sudoku puzzle
__global__ void solveSudoku(char* board, bool* solutionFound) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= N * N || *solutionFound) return; // Out of bounds check or solution already found

    while (!(*solutionFound)) {
        if (idx >= N * N) return; // Out of bounds check

        int row = idx / N;
        int col = idx % N;

        if (board[idx] == '.') {
            bool localSolutionFound = false;
            for (char num = '1'; num <= '9'; ++num) {
                if (isValid(board, row, col, num)) {
                    board[idx] = num;
                    localSolutionFound = true;
                    break;
                }
            }
            if (!localSolutionFound) {
                board[idx] = '.';
                return;
            }
        }

        if (idx == N * N - 1) {
            *solutionFound = true;
            return; // Sudoku solved
        }

        idx++;
    }
}

// Host function to print the Sudoku board
// Function: printSudoku
// Description: Prints the solved Sudoku board to the console.
// Parameters:
// - board: The solved Sudoku board represented as a 2D vector of characters.
void printSudoku(const std::vector<char>& board) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << board[i * N + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Sudoku board
    std::vector<char> board = {
        '5','3','.','.','7','.','.','.','.',
        '6','.','.','1','9','5','.','.','.',
        '.','9','8','.','.','.','.','6','.',
        '8','.','.','.','6','.','.','.','3',
        '4','.','.','8','.','3','.','.','1',
        '7','.','.','.','2','.','.','.','6',
        '.','6','.','.','.','.','2','8','.',
        '.','.','.','4','1','9','.','.','5',
        '.','.','.','.','8','.','.','7','9'
    };

    char* dev_board;
    bool* dev_solutionFound;

    GPUErrorAssertion(cudaMalloc((void**)&dev_board, N * N * sizeof(char)));
    GPUErrorAssertion(cudaMalloc((void**)&dev_solutionFound, sizeof(bool)));

    GPUErrorAssertion(cudaMemcpy(dev_board, board.data(), N * N * sizeof(char), cudaMemcpyHostToDevice));

    bool solutionFound = false;

    auto start = std::chrono::steady_clock::now(); // Start time measurement

    solveSudoku<<<1, N * N>>>(dev_board, dev_solutionFound);
    GPUErrorAssertion(cudaGetLastError()); // Check for kernel launch errors

    GPUErrorAssertion(cudaMemcpy(&solutionFound, dev_solutionFound, sizeof(bool), cudaMemcpyDeviceToHost));

    auto end = std::chrono::steady_clock::now(); // End time measurement
    std::chrono::duration<double> elapsed = end - start;

    if (solutionFound) {
        std::vector<char> solvedBoard(N * N);
        GPUErrorAssertion(cudaMemcpy(solvedBoard.data(), dev_board, N * N * sizeof(char), cudaMemcpyDeviceToHost));
        printSudoku(solvedBoard);
        std::cout << "Elapsed time for GPU solving: " << elapsed.count() << " seconds" << std::endl;
    } else {
        std::cout << "No solution exists for the given Sudoku board.\n";
    }

    GPUErrorAssertion(cudaFree(dev_board));
    GPUErrorAssertion(cudaFree(dev_solutionFound));

    return 0;
}
