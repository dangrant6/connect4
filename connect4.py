import pygame
import numpy as np
import random
import sys
import math
from copy import deepcopy

#colors for board
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 42, 243)
BLACK = (0, 0, 0)

#chip beginning states
EMPTY_SLOT = 0
PLAYER_CHIP = 1
AI_CHIP = 2

#player and AI turns
PLAYER = 0
AI = 1


class ConnectFour:
    #board attributes
    def __init__(self, rows=6, columns=7):
        self.rows = rows
        self.columns = columns
        self.board = np.zeros((rows, columns))

    def drop_chip(self, row, col, chip):

        #drop chip in hole
        self.board[row][col] = chip

    def is_valid_location(self, col):

        #check to see if column has any empty slots
        return self.board[self.rows-1][col] == 0

    def get_empty_row(self, col):

        #return empty row
        #Note that the topmost empty row will be returned
        #This will be rectified due to the way the board is printed in function below

        for row in range(self.rows):
            if self.board[row][col] == 0:
                return row

    def print_board(self):

        #The board is printed in reverse order (from bottom row to top)
        #due to the nature of the indexing utilized
        print(np.flip(self.board, 0))

    def get_valid_positions(self):
        
        #determine which columns a chip can be dropped into by players
        valid_positions = []

        for col in range(self.columns):
            if self.is_valid_location(col):
                valid_positions.append(col)

        return valid_positions

    def winning_move(self, chip):

        #check horizontal locations for win
        #all columns except the last three are checked because these are the only possible starting
        #positions for a winning move that lead to four horizontal chip
        for col in range(self.columns-3):

            #iterating over all the rows
            for row in range(self.rows):
                if self.board[row][
                        col] == chip and self.board[row][col+1] == chip and self.board[
                        row][col+2] == chip and self.board[row][col+3] == chip:
                    return True

        #check vertical locations for win
        for col in range(self.columns):

            #only the first three rows are checked since the starting position for a winning move
            #can only be from those rows to have a line of four
            for row in range(self.rows-3):
                if self.board[row][
                        col] == chip and self.board[row+1][
                        col] == chip and self.board[row+2][col] == chip and self.board[row+3][col] == chip:
                    return True

        # check positive diagonal locations for a winning configuration

        #starting position for a winning move are all rows and columns except the last three of each
        for col in range(self.columns-3):
            for row in range(self.rows-3):
                if self.board[row
                              ][col] == chip and self.board[row+1
                                                            ][col+1] == chip and self.board[row+2][col+2] == chip and self.board[row+3][col+3] == chip:
                    return True

        # check negative diagonal locations for a winning configuration

        #starting position for a winning move are all rows and columns except the last three columns
        #and first three rows
        for col in range(self.columns-3):
            for row in range(3, self.rows):
                if self.board[row][col] == chip and self.board[row-1][col+1] == chip and self.board[row-2][col+2] == chip and self.board[row-3][col+3] == chip:
                    return True

    def evaluate_window(self, window, chip):
        #assigning scores
        score = 0

        #if the current chip is the player's, set opposing chip to AI chip
        if chip == PLAYER_CHIP:
            opp_chip = AI_CHIP
        else:
            opp_chip = PLAYER_CHIP

        #if a player has successfully connected four chips, add 100 to score
        if window.count(chip) == 4:
            score += 100

        #if a player has successfully connected three chips, add 100 to score
        elif window.count(chip) == 3 and window.count(EMPTY_SLOT) == 1:
            score += 5

        #if a player has successfully connected two chips, add 100 to score
        elif window.count(chip) == 2 and window.count(EMPTY_SLOT) == 2:
            score += 2

        #if the opposing player has successfully connected three chips, subtract 4 from score
        if window.count(opp_chip) == 3 and window.count(EMPTY_SLOT) == 1:
            score -= 4

        #return the score given the input window
        return score

    def score_position(self, chip, window_length=4):
        #score player based on board configuration
        score = 0

        #center column
        center_window = [int(i) for i in list(self.board[:, self.columns//2])]
        score += center_window.count(chip) * 3

        #rows
        for row in range(self.rows):

            #get all the columns of the particular row
            row_window = [int(i) for i in list(self.board[row, :])]

            #for all the columns except the last three in the given row
            for col in range(self.columns-3):

                #calculate the score using the scoring system
                window = row_window[col:col+window_length]
                score += self.evaluate_window(window, chip)

        #score verticals (column chips configurations)
        for col in range(self.columns):
            col_window = [int(i) for i in list(self.board[:, col])]
            for row in range(self.rows-3):
                window = col_window[row:row+window_length]
                score += self.evaluate_window(window, chip)

        #score positive diagonals
        for row in range(self.rows-3):
            for col in range(self.columns-3):
                window = [self.board[row+i][col+i]
                          for i in range(window_length)]
                score += self.evaluate_window(window, chip)

        #score negative diagonals
        for row in range(self.rows-3):
            for col in range(self.columns-3):
                window = [self.board[row+3-i][col+i]
                          for i in range(window_length)]
                score += self.evaluate_window(window, chip)

        return score

    def terminal_node(self):
        #check if a terminal state has been reached

        #check if a player has connected 4 or
        #if there are no other empty slots on the board
        return self.winning_move(PLAYER_CHIP) or self.winning_move(AI_CHIP) or len(self.get_valid_positions()) == 0

    def alpha_beta_minimax_search(self, depth, alpha, beta, maximizingPlayer):
        #evaluate the best branches possible going as 
        #deep down the tree as specified by depth

        #determine possible columns that can played
        possible_moves = self.get_valid_positions()

        if depth == 0 or self.terminal_node():
            if self.terminal_node():
                if self.winning_move(AI_CHIP):
                    return (None, 50000)
                elif self.winning_move(PLAYER_CHIP):
                    return (None, -50000)
                else:  #game over
                    return (None, 0)
            else:  #depth is zero
                #return board heurisitic value
                return (None, self.score_position(AI_CHIP))

        if maximizingPlayer:
            value = -float("inf")
            move = np.random.choice(possible_moves)

            for col in possible_moves:
                row = self.get_empty_row(col)
                next_state = deepcopy(self)
                next_state.drop_chip(row, col, AI_CHIP)
                new_score = next_state.alpha_beta_minimax_search(
                    depth-1, alpha, beta, False)[1]

                #update value to highest score encountered based on all explored moves
                if new_score > value:
                    value = new_score
                    move = col

                alpha = max(value, alpha)

                if alpha >= beta:
                    break

            return move, value

        #minimizing player trying to obtain the lowest score
        else:

            value = float("inf")
            move = np.random.choice(possible_moves)

            for col in possible_moves:
                row = self.get_empty_row(col)
                next_state = deepcopy(self)
                next_state.drop_chip(row, col, PLAYER_CHIP)
                new_score = next_state.alpha_beta_minimax_search(
                    depth-1, alpha, beta, True)[1]

                #update value to highest score encountered based on all explored moves
                if new_score < value:
                    value = new_score
                    move = col

                beta = min(value, beta)

                if alpha >= beta:
                    break

            return move, value


def draw_pygame_board(board):
    """
    draw the board in pygame 

    Input:
    - board: A ConnectFour class which contains an attribute for the board and possible actions

    """

    #iterating over all the columns of the inputted board
    for col in range(board.columns):

        #iterating over all the rows of the inputted board
        for row in range(board.rows):

            #draw a rectangle and a circle inside it to build up a visualization of an empty ConnectFour board
            pygame.draw.rect(screen, BLUE, (col*TILESIZE, row *
                                            TILESIZE+TILESIZE, TILESIZE, TILESIZE))
            pygame.draw.circle(screen, BLACK, (int(
                col*TILESIZE+TILESIZE/2), int(row*TILESIZE+TILESIZE+TILESIZE/2)), RADIUS)

    #iterate through every row and column
    for col in range(board.columns):
        for row in range(board.rows):

            #if the chip in the current board position is the player's, replace with a red chip
            if board.board[row][col] == PLAYER_CHIP:
                pygame.draw.circle(screen, RED, (int(
                    col*TILESIZE+TILESIZE/2), height-int(row*TILESIZE+TILESIZE/2)), RADIUS)

            #if the chip in the current board position is the AI's, replace with a yellow chip
            elif board.board[row][col] == AI_CHIP:
                pygame.draw.circle(screen, YELLOW, (int(
                    col*TILESIZE+TILESIZE/2), height-int(row*TILESIZE+TILESIZE/2)), RADIUS)
    pygame.display.update()


#initialize board and print the empty board
board = ConnectFour(6, 7)

#initialize game_over to false to signal that the game has started/in progress
game_over = False

pygame.init()

#initialize attributes for the pygame board
TILESIZE = 100
width = board.columns * TILESIZE
height = (board.rows+1) * TILESIZE

#size of pygame window
size = (width, height)

#radius size for player chips given size of squares
RADIUS = int(TILESIZE/2 - 5)

screen = pygame.display.set_mode(size)

draw_pygame_board(board)
pygame.display.update()

pygame.font.init()
game_font = pygame.font.SysFont('helvetica', 75)

#randomly choose who starts first
turn = random.randint(PLAYER, AI)

while not game_over:

    #if the game is quit, exit
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    #if the mouse is moved, move the chip along with it
    if event.type == pygame.MOUSEMOTION:
        pygame.draw.rect(screen, BLACK, (0, 0, width, TILESIZE))
        xpos = event.pos[0]
        if turn == PLAYER:
            pygame.draw.circle(screen, RED, (xpos, int(TILESIZE/2)), RADIUS)

    #update the display to see changes
    pygame.display.update()

    #if the mouse is clicked (to drop)
    if event.type == pygame.MOUSEBUTTONDOWN:
        pygame.draw.rect(screen, BLACK, (0, 0, width, TILESIZE))

        # if it is the player's turn
        if turn == PLAYER:

            #get the drop position and use it to estimate the intended board column
            xpos = event.pos[0]
            col = int(math.floor(xpos/TILESIZE))

            #if the selected column is a valid column
            if board.is_valid_location(col):

                #drop the chip down into the next available row
                row = board.get_empty_row(col)
                board.drop_chip(row, col, PLAYER_CHIP)

                #if the drop results in successfully connecting four chips
                if board.winning_move(PLAYER_CHIP):
                    #player wins
                    label = game_font.render("You win(how???!)!", 1, RED)
                    screen.blit(label, (40, 10))
                    game_over = True

                turn += 1
                turn = turn % 2

                #board.print_board()
                draw_pygame_board(board)

        #ask for AI input
        if turn == AI and not game_over:

            col, minimax_score = board.alpha_beta_minimax_search(
                4, -float('inf'), float('inf'), True)

            if board.is_valid_location(col):
                row = board.get_empty_row(col)
                board.drop_chip(row, col, AI_CHIP)

                if board.winning_move(AI_CHIP):
                    label = game_font.render("AI wins!", 1, YELLOW)
                    screen.blit(label, (40, 10))
                    game_over = True

                draw_pygame_board(board)

                turn += 1
                turn = turn % 2
