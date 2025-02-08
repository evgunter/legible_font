import os
import json
import numpy as np

# TODO: this is only used in display_boards, which is only used when running this file,
# not when importing things from it. it might be nicer to not import it in that case
import matplotlib.pyplot as plt

BOARDS_PATH = os.path.join(os.path.dirname(__file__), "../../test_data/boards.txt")

def decode_board(board_int, n, m):
    """decode board_int from an int to a binary n x m matrix"""
    matrix = np.array([[False] * m for _ in range(n)])
    for i in range(n):
        for j in range(m):
            matrix[i][j] = board_int & 1 == 1
            board_int >>= 1
    return matrix

def display_boards(boards):
    """display boards in a grid"""    
    height = int(len(boards)**0.5)
    width = len(boards) // height + 1
    fig, axes = plt.subplots(height, width, figsize=(height/2, width/2))
    for i, ax in enumerate(axes.flatten()):
        ax.axis("off")
        if i >= len(boards):
            continue
        ax.imshow(boards[i], cmap="gray")
    plt.show()

def get_boards_bool():
    """load the boards from the file and return them as a list of n x m binary matrices where true is black and false is white"""
    with open(BOARDS_PATH, "r") as f:
        dims_str, boards_str = f.read().splitlines()
        n, m = json.loads(dims_str)
        boards = json.loads(boards_str)
    return np.array([decode_board(board, n, m) for board in boards]), (n, m)

def get_boards():
    """load the boards from the file and return them as a list of n x m float matrices"""
    bool_boards, (n, m) = get_boards_bool()
    # true is black and false is white, so we need to reverse the automatic bool -> int conversion
    return 1 - bool_boards.astype(float), (n, m)


if __name__ == "__main__":
    boards, _ = get_boards()
    display_boards(boards)
