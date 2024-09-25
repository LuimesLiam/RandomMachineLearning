import chess
import chess.svg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from IPython.display import clear_output, display
from PIL import Image
import io
import time
import os
import cairosvg

#check cuda 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"USING device: {device}")


GAMMA = 0.99  
EPSILON_START = 1.0  
EPSILON_END = 0.05  
EPSILON_DECAY = 0.995  
LEARNING_RATE = 0.001  
BATCH_SIZE = 64  
MEMORY_SIZE = 10000  
TARGET_UPDATE_FREQ = 10  
SAVE_MODEL_FREQ = 100  


class DQN(nn.Module):
    def __init__(self):
        super(DQN,self).__init__()
        self.fc1 = nn.Linear(64,128) 
        self.fc2 = nn.Linear(128,128) 
        self.fc3 = nn.Linear(128,1) 
    
    def forward(self,x):
        x = torch.relu(self.fc1(x)) 
        x = torch.relu(self.fc2(x)) 
        return self.fc3(x) 


class ReplayMemory:
    def __init__(self,capacity):
        self.memory = deque(maxlen=capacity) 
    
    def push(self,transition):
        self.memory.append(transition)

    def sample(self,batch_size):
        return random.sample(self.memory,batch_size)

    def __len__(self):
        return len(self.memory)
    

def get_piece_value(piece):
    if piece is None:
        return 0
    piece_values ={
        chess.PAWN: 1, 
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5, 
        chess.QUEEN: 9,
        chess.KING: 100
    }
    return piece_values[piece.piece_type] * (1 if piece.color==chess.WHITE else -1)


def get_state(board):
    state =np.zeros(64,dtype=np.float32) 

    for i, piece in enumerate(board.piece_map()):
        state[i] = get_piece_value(board.piece_at(piece))
    return torch.tensor(state,dtype=torch.float32).to(device) 


def get_reward(board):
    if board.is_checkmate():
        return 1 if board.turn ==chess.BLACK else -1
    return 0

def select_best_action(policy_net, state,board):
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


def optimize_model(policy_net, target_net, optimizer, memory):
    if (len(memory)<BATCH_SIZE):
        return
    transitions = memory.sample(BATCH_SIZE)
    batch_state, batch_action, batch_reward, batch_next_state,batch_done = zip(*transitions)

    batch_next_state = torch.stack(batch_next_state).to(device)
    batch_state = torch.stack(batch_state).to(device) 
    
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device) 
    batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

    q_values = policy_net(batch_state) 
    next_q_values = target_net(batch_next_state).detach() 

    expected_q_values = batch_reward + (GAMMA * next_q_values *(1-batch_done)) 
    loss = nn.MSELoss()(q_values,expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


img_num = 0

def display_board(board, episode, move_count):
    global img_num
    clear_output(wait=True)
    svg_board = chess.svg.board(board=board)
    png_bytes = cairosvg.svg2png(bytestring=svg_board.encode('utf-8'))
    img = Image.open(io.BytesIO(png_bytes))
    img.save(f"PythonChess/simpleChess/chess_game_ep{img_num}_move.png")
    display(img)
    print(f"Episode: {episode}, Move: {move_count}")
    img_num += 1
    if img_num > 5:
        img_num = 0
    time.sleep(0.5)

def save_model(model, episode, directory='PythonChess/saved_models'):
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(model.state_dict(), f"{directory}/chess_model_episode_{episode}.pth")
    print(f"Model saved at episode {episode}")
def load_model(model_path):
    model = DQN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def train(num_episodes=1000, visualize=False, model_path= None):
    if (model_path != None):
        policy_net = load_model(model_path)
    else:
        policy_net = DQN().to(device)
    target_net = DQN().to(device)

    target_net.load_state_dict(policy_net.state_dict()) 
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(),lr=LEARNING_RATE)
    memory = ReplayMemory(MEMORY_SIZE)

    epsilon = EPSILON_START

    for episode in range(num_episodes):
        board = chess.Board()
        state = get_state(board) 
        total_reward = 0
        move_count = 0

        while not board.is_game_over():
            if (random.random() < epsilon): 
                action = random.choice(list(board.legal_moves))
            else:
                action = select_best_action(policy_net, state,board)
            board.push(action)
            next_state = get_state(board)
            reward = get_reward(board)

            memory.push((state,action,reward,next_state,board.is_game_over())) 

            state = next_state 
            total_reward += reward
            move_count +=1

            optimize_model(policy_net, target_net, optimizer, memory)

            if visualize and episode > 400: 
                display_board(board,episode,move_count)
            if board.is_game_over():
                break

        if episode % SAVE_MODEL_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % SAVE_MODEL_FREQ == 0 or episode == num_episodes -1:
            save_model(policy_net,episode)

        epsilon = max(EPSILON_END, epsilon*EPSILON_DECAY)
        
        
        print(f"Episode {episode}, Total Reward: {total_reward}, Moves: {move_count}")

    save_model(policy_net, 'final')


if __name__ == "__main__":
    train(num_episodes=1000, visualize=False,model_path="saved_models/chess_model_episode_400.pth" )