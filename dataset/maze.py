import random
import pandas as pd
import os
import pickle
import numpy as np
import torch


class RecursiveBackMaze:
    def __init__(self, width, height):

        self.width = width // 2 * 2 + 1
        self.height = height // 2 * 2 + 1

        # this creates a 2d-array for your maze data (0: path, 1: wall)
        self.cells = [
            [1 for x in range(self.width)]
            for y in range(self.height)
        ]

    def set_path(self, x, y):
        self.cells[y][x] = 0

    def set_wall(self, x, y):
        self.cells[y][x] = 1

    # a function to return if the current cell is a wall,
    #  and if the cell is within the maze bounds
    def is_wall(self, x, y):
        # checks if the coordinates are within the maze grid
        if 0 <= x < self.width and 0 <= y < self.height:
            # if they are, then we can check if the cell is a wall
            return self.cells[y][x]
        # if the coordinates are not within the maze bounds, we don't want to go there
        else:
            return 0

    def create_maze_rec(self, x, y):
        # set the current cell to a path, so that we don't return here later
        self.set_path(x, y)
        # we create a list of directions (in a random order) we can try
        all_directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        random.shuffle(all_directions)

        # we keep trying the next direction in the list, until we have no directions left
        while len(all_directions) > 0:

            # we remove and return the last item in our directions list
            direction_to_try = all_directions.pop()

            # calculate the new node's coordinates using our random direction.
            # we *2 as we are moving two cells in each direction to the next node
            node_x = x + (direction_to_try[0] * 2)
            node_y = y + (direction_to_try[1] * 2)

            # check if the test node is a wall (eg it hasn't been visited)
            if self.is_wall(node_x, node_y):
                # success code: we have found a path

                # set our linking cell (between the two nodes we're moving from/to) to a path
                link_cell_x = x + direction_to_try[0]
                link_cell_y = y + direction_to_try[1]
                self.set_path(link_cell_x, link_cell_y)

                # "move" to our new node. remember we are calling the function every
                #  time we move, so we call it again but with the updated x and y coordinates
                self.create_maze_rec(node_x, node_y)
        return

    def create_maze(self, x, y):
        self.create_maze_rec(x, y)
        # Set entrance and exit
        self.cells[x][y] = 2

        for i in range(self.width - 1, 0, -1):
            if (self.cells[self.height - 2][i] == 0):
                self.cells[self.height - 2][i] = 3
                break
        return maze


def print_maze_cute(maze):
    string = ""
    conv = {
        3: "E ",
        2: "S ",
        1: "██",
        0: "  "
    }
    for y in range(len(maze)):
        for x in range(len(maze[y])):
            string += conv[maze[y][x]]
        string += "\n"
    print(string)


height = 63
width = 63
res = []
# maze = prim_generate_labyrinth(width, height)
for i in range(10000):
    maze = RecursiveBackMaze(width=width, height=height)
    res.append(maze.create_maze(1, 1).cells)
    # print_maze_cute(maze.cells)
    # print(maze.cells)
    # calculate_maze_complexity(maze.cells) / sqrt(width * height)
# for m in res:
#   print_maze_cute(m)

res = np.array(res, dtype=np.float32)

outname = 'dataset.pkl'
outdir = 'data/maze'
if not os.path.exists(outdir):
    os.mkdir(outdir)

fullname = os.path.join(outdir, outname)

# pd.DataFrame(res).to_csv(fullname, ';')
with open(fullname, 'wb') as f:
    pickle.dump(res, f)
