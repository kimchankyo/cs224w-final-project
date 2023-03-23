import os
import torch
from pathlib import Path
from torch_geometric.datasets import MNISTSuperpixels
from models import GNN
from pipeline import Trainer


if __name__ == '__main__':
  root = str(Path(os.getcwd()).parent / 'data' / 'mnist-superpixels')
  dataset = MNISTSuperpixels(root=root)
  
  model = GNN(dataset.num_features, 64, dataset.num_classes, numLayers=4, dropout=0.2)
  optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

  trainer = Trainer(str(Path(os.getcwd()).parent), 'graphsage', model, optimizer, replace=True)
  trainer.train(dataset)