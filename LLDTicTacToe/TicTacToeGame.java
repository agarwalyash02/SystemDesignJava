package SystemDesignJava.LLDTicTacToe;

import java.util.Deque;
import java.util.LinkedList;
import java.util.List;
import java.util.Scanner;

import SystemDesignJava.LLDTicTacToe.Model.Board;
import SystemDesignJava.LLDTicTacToe.Model.PieceType;
import SystemDesignJava.LLDTicTacToe.Model.Player;
import SystemDesignJava.LLDTicTacToe.Model.PlayingPiece;
import SystemDesignJava.LLDTicTacToe.Model.PlayingPieceO;
import SystemDesignJava.LLDTicTacToe.Model.PlayingPieceX;

public class TicTacToeGame {
    Deque<Player> players;
    Board gameBoard;

    public void intializeGame() {
        // creating 2 players
        players = new LinkedList<>();

        PlayingPieceX crossPiece = new PlayingPieceX();
        Player player1 = new Player("player1", crossPiece);

        PlayingPieceO noughtsPiece = new PlayingPieceO();
        Player player2 = new Player("player2", noughtsPiece);

        players.add(player1);
        players.add(player2);

        gameBoard = new Board(3);
    }

    public String startGame() {
        boolean noWinner = true;
        while (noWinner) {
            // take out the player whose turn is and also put the player in the list back
            Player playerTurn = players.removeFirst();

            // get the free space from the board
            gameBoard.printBoard();
            List<Pair<Integer, Integer>> freeSpaces = gameBoard.getFreeCells();
            if (freeSpaces.isEmpty) {
                noWinner = false;
                continue;
            }

            // read the user input
            System.out.print("Player: " + playerTurn.getName() + " Enter row, column: ");
            Scanner inputScanner = new Scanner(System.in);
            String s = inputScanner.nextLine();
            String[] values = s.split(",");
            int inputRow = Integer.parseInt(values[0]);
            int inputColummn = Integer.parseInt(values[1]);

            // place the piece
            boolean pieceAddedSuccessfully = gameBoard.addPiece(inputRow, inputColummn, playerTurn.getPlayingPiece());
            if (!pieceAddedSuccessfully) {
                System.out.println("Incorrect position chosen, try again");
                players.addFirst(playerTurn);
                continue;
            }
            players.addLast(playerTurn);
            boolean winner = isThereWinner(inputRow, inputColummn, playerTurn.getPlayingPiece());
            if (winner) {
                return playerTurn.getName();
            }

            return "tie!";
        }
    }

    public boolean isThereWinner(int row, int column, PlayingPiece playingPiece) {
        boolean rowMatch = true;
        boolean columnMatch = true;
        boolean diagonalMatch = true;
        boolean antiDiagonalMatch = true;

        // need to check in row
        for (int i = 0; i < gameBoard.size; i++) {
            if (gameBoard.board[row][i] == null || gameBoard.board[row][i] != playingPiece) {
                rowMatch = false;
            }
        }

        // need to check in column
        for (int i = 0; i < gameBoard.size; i++) {
            if (gameBoard.board[i][column] == null || gameBoard.board[i][column] != playingPiece) {
                columnMatch = false;
            }
        }

        // need to check the diagonals
        for (int i = 0, j = 0; i < gameBoard.size; i++, j++) {
            if (gameBoard.board[i][j] == null || gameBoard.board[i][j] != playingPiece) {
                diagonalMatch = false;
            }
        }

        // need to check the anti-diagonal
        for (int i = 0, j = gameBoard.size - 1; i < gameBoard.size; i++, j--) {
            if (gameBoard.board[i][j] == null || gameBoard.board[i][j] != playingPiece) {
                antiDiagonalMatch = false;
            }
        }
        return rowMatch || columnMatch || diagonalMatch || antiDiagonalMatch;
    }
}
