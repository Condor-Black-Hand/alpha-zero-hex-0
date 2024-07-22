from Coach import Coach
from hex.HexGame import HexGame as Game
from hex.pytorch.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 1000,
    'numEps': 100,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200000,
    'numMCTSSims': 75,
    'arenaCompare': 40,
    'cpuct': 1.5,

    'checkpoint': './pretrained_models/hex/',
    'load_model': True,
    'load_folder_file': ('./pretrained_models/hex/','checkpoint_396.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})

if __name__=="__main__":
    g = Game(11)
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
