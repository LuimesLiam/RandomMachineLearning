import chess
import torch
import torch.nn as nn
import numpy as np

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def get_state(board):
    state = np.zeros(64, dtype=np.float32)
    for i, piece in enumerate(board.piece_map()):
        state[i] = get_piece_value(board.piece_at(piece))
    return torch.tensor(state, dtype=torch.float32).to(device)

def get_piece_value(piece):
    if piece is None:
        return 0
    piece_values = {
        chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 100
    }
    return piece_values[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)

def select_best_action(policy_net, state, board):
    with torch.no_grad():
        best_action = None
        max_q_value = -float('inf')
        for action in board.legal_moves:
            board.push(action)
            next_state = get_state(board)
            q_value = policy_net(next_state).item()
            if q_value > max_q_value:
                max_q_value = q_value
                best_action = action
            board.pop()
        return best_action

def load_model(model_path):
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def play_game(policy_net):
    board = chess.Board()
    while not board.is_game_over():
        print(board)
        if board.turn == chess.WHITE:
            # Human's turn (White)
            move = input("Enter your move (e.g., e2e4): ")
            try:
                board.push_san(move)
            except ValueError:
                print("Invalid move. Try again.")
                continue
        else:
            # AI's turn (Black)
            state = get_state(board)
            action = select_best_action(policy_net, state, board)
            print(f"AI plays: {board.san(action)}")
            board.push(action)

        print()

    result = board.result()
    print("Game over!")
    print(f"Result: {result}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "PythonChess/saved_models/chess_model_episode_400.pth"  # Update with your model path
    policy_net = load_model(model_path)
    play_game(policy_net)
