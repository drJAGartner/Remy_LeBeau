# Remy LeBeau
A chess AI with an Neural Network engine written in PyTorch.

## Overview
The approach was to make a Neural Network that takes a simple vector representation of the board and creates a feed forward network who's purpose is to predict the probability of winning the game based on the current position.  

## Dependency
Many chess engines have the network try to "learn" the rules of chess.  To short circuit this, I only have the engine evaluate the positions that are leagal.  Since there is an excellent engine for doing this already, I leveraged the `python-chess` package:

https://pypi.org/project/python-chess/
