package SystemDesignJava.LLDTicTacToe.Model;

public abstract class PlayingPiece {
    public PieceType pieceType;

    PlayingPiece(PieceType pieceType) {
        this.pieceType = pieceType;
    }
}
