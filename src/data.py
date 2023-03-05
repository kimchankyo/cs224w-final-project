from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_networkx
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import os



if __name__ == "__main__":
  root = str(Path(os.getcwd()).parent / 'data' / 'mnist-superpixels')
  dataset = MNISTSuperpixels(root=root)
  data = dataset[0]
  print(data.y)
  G = to_networkx(data, to_undirected=True)
  nx.draw_networkx(G, with_labels=False)
  plt.show()