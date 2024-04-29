import java.util.concurrent.ThreadLocalRandom;

public class Board {
    Cell[][] cells;

    Board(int boardSize, int numOfsnakes, int numOfLadders) {
        initializeCells(boardSize);
        addSnakesLadders(cells, numOfsnakes, numOfLadders);
    }

    void initializeCells(int boardSize) {
        cells = new Cell[boardSize][boardSize];
        for (int i = 0; i < boardSize; i++) {
            for (int j = 0; j < boardSize; j++) {
                Cell cellObj = new Cell();
                cells[i][j] = cellObj;
            }
        }
    }

    void addSnakesLadders(Cell[][] cells, int numOfsnakes, int numOfladder) {
        while (numOfsnakes > 0) {
            int snakeStart = ThreadLocalRandom.current().nextInt(1, cells.length * cells.length - 1);
            int snakeEnd = ThreadLocalRandom.current().nextInt(1, cells.length * cells.length - 1);
            if (snakeStart <= snakeEnd) {
                continue;
            }

            Jump snakeObj = new Jump();
            snakeObj.start = snakeStart;
            snakeObj.end = snakeEnd;
            Cell currentCell = getCell(snakeStart);
            currentCell.jump = snakeObj;
            numOfsnakes--;
        }
        while (numOfladder > 0) {
            int laddertart = ThreadLocalRandom.current().nextInt(1, cells.length * cells.length - 1);
            int snakeEnd = ThreadLocalRandom.current().nextInt(1, cells.length * cells.length - 1);
            if (laddertart <= snakeEnd) {
                continue;
            }

            Jump snakeObj = new Jump();
            snakeObj.start = laddertart;
            snakeObj.end = snakeEnd;
            Cell currentCell = getCell(laddertart);
            currentCell.jump = snakeObj;
            numOfladder--;
        }
    }

    Cell getCell(int playerPosition) {
        int boardRow = playerPosition / cells.length;
        int boardColumn = playerPosition % cells.length;
        return cells[boardRow][boardColumn];
    }
}
