from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.utils import to_networkx
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_tar
from torch_geometric.utils.convert import from_networkx

from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple
from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import center_of_mass
import networkx as nx
import matplotlib.pyplot as plt
import os
import os.path as osp
import numpy as np

from torchvision import datasets
import torch
import torchvision
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings("ignore")


# Download MNISTSuperpixels Dataset from PyTorch Geometric
def downloadMNISTSuperpixels(
    dataRoot: str = str(Path(os.getcwd()).parent / 'data'), 
    name: str = 'mnist-superpixels', visualize: bool = False
  ) -> None:
  dataset = MNISTSuperpixels(root=str(Path(dataRoot) / name))
  if visualize:
    data = dataset[0]
    G = to_networkx(data, to_undirected=True)
    nx.draw_networkx(G, with_labels=False)
    plt.show()

# Superpixel Generation Functions
def segmentSuperpixels(image: np.ndarray, numSP: int, compactness: int, 
                       maxIter: int = 500) -> None:
  """
  compactness (float): Balances color proximity and space proximity. 
                       Higher values give more weight to space proximity, 
                       making superpixel shapes more square/cubic. In SLICO 
                       mode, this is the initial compactness. This parameter 
                       depends strongly on image contrast and on the shapes 
                       of objects in the image. We recommend exploring 
                       possible values on a log scale, e.g., 0.01, 0.1, 1, 
                       10, 100, before refining around a chosen value.
  """
  image = image.astype(np.float32) / 255.0    # Normalization

  superpix = slic(image, n_segments=numSP, compactness=1, 
                  channel_axis=2, start_label=0)
  assert len(np.unique(superpix)) > 1

  # SLIC Refinement
  iter = 0
  areas = np.array([np.sum(superpix == l) for l in np.unique(superpix)])
  refineSegments = 2
  while len(np.unique(superpix)) < numSP:
    maxAreaLabel = np.argmax(areas)                   # Find largest area
    mask = np.where(superpix == maxAreaLabel, 1, 0)   # Generate mask
    
    # Try segmenting into two pieces
    segment = slic(image, n_segments=refineSegments, compactness=compactness, 
                   channel_axis=2, start_label=0, mask=mask)
    numSegments = len(np.unique(segment)[1:])
    if numSegments < 2:
      refineSegments += 1
      continue
    elif numSegments > 2:
      # Merge segments until 2 segments remain
      segmentAreas = np.array([np.sum(segment == l) \
                               for l in np.unique(segment)[1:]])
      sortedLabels = np.argsort(segmentAreas)
      
      # Merge the smallest 2 segments (larger label is consumed)
      l1, l2 = np.sort(sortedLabels[:2])
      segment = np.where(segment == l2, l1, segment)
      segment = np.where(segment > l2, segment-1, segment)
    superpix = np.where(superpix > maxAreaLabel, superpix+1, superpix)

    superpix = np.where(segment == 1, maxAreaLabel+1, superpix)
    areas = np.array([np.sum(superpix == l) for l in np.unique(superpix)])
    if iter == maxIter: break
    iter += 1
  
  return superpix

def extractNodeFeatures(image: np.ndarray, numSP: int, superpix: np.ndarray, 
                        shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
  indices = np.random.permutation(numSP) if shuffle else np.arange(numSP)
  ordering = np.unique(superpix)[indices].astype(np.int32)
  
  image = image[:, :, None] if len(image.shape) == 2 else image   # 2D -> 3D
  numChannels = 1 if image.shape[2] == 1 else 3

  features = []
  for i in ordering:
    mask = np.squeeze((superpix == i))
    avg = np.zeros(numChannels)
    for c in range(numChannels):
      avg[c] = np.mean(image[:, :, c][mask])
    center = np.array(center_of_mass(mask))
    features.append(np.hstack((avg, center)))
  features = np.array(features).astype(np.float32)
  return features[:, :3], features[:, 3:]

def extractEdges(superpix: np.ndarray, fullConnect: bool) -> List[Tuple[int, int]]:
  edges = []
  if fullConnect:
    nodes = np.unique(superpix)
    for i in range(nodes.shape[0]):
      for j in range(nodes.shape[0]):
        if i != j: edges.append((nodes[i], nodes[j]))
  else:
    for node in np.unique(superpix):
      # Find border mask
      H, W = superpix.shape
      border = np.zeros(superpix.shape)
      for i in range(H):
        for j in range(W):
          pix = superpix[i, j]
          if i < H-1:
            down = superpix[i+1, j]
            if pix == node and down != node: border[i+1, j] = 1
          if j < W-1:
            right = superpix[i, j+1]
            if pix == node and right != node: border[i, j+1] = 1
          if i < H-1 and j < W-1:
            if pix != node and (right == node or down == node): border[i, j] = 1
      neighbors = np.unique(np.where(border == 1, superpix, -1))
      neighbors = neighbors[1:] if -1 in neighbors else neighbors
      for n in neighbors:
        edges.append((node, n))
  return edges

def getDataList(images: np.ndarray, labels: np.ndarray, numSP: int, 
                compactness: float, shuffle: bool = True, fullConnect: bool = False):
  dataList = []
  for i in tqdm(range(images.shape[0])):
    image, y = images[i], labels[i]

    superpix = segmentSuperpixels(image, numSP, compactness)

    # Node Feature Construction
    x, pos = extractNodeFeatures(image, numSP, superpix, shuffle)
    
    # Edge Construction
    edges = extractEdges(superpix, fullConnect)

    # Graph Construction
    G = nx.Graph()
    G.add_nodes_from(np.arange(x.shape[0]))
    G.add_edges_from(edges)
    
    # Convert to PyTorch Geometric data
    data = Data(torch.Tensor(x), from_networkx(G).edge_index, 
                None, torch.Tensor([y]).type(torch.int64), torch.Tensor(pos))
    dataList.append(data)
  return dataList


class CIFAR10Superpixels(InMemoryDataset):

  url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

  def __init__(self, root, transform=None, pre_transform=None, pre_filter=None,
               train: bool = True, numSP: int = 75, compactness: float = 5.0, 
               shuffle: bool = True, fullConnect: bool = False):
    self.train = train
    self.numSP = numSP
    self.compactness = compactness
    self.isShuffle = shuffle
    self.fullConnect = fullConnect
    super().__init__(root, transform, pre_transform, pre_filter)
    path = self.processed_paths[0] if train else self.processed_paths[1]
    self.data, self.slices = torch.load(path)

  @property
  def raw_file_names(self):
    return 'cifar-10-python.tar.gz'

  @property
  def processed_file_names(self):
    return ['train_data.pt', 'test_data.pt']
  
  def download(self):
    download_url(self.url, self.raw_dir)

  def process(self):
    processed_path = self.processed_paths[0] if self.train else self.processed_paths[1]

    if not os.path.exists(processed_path):
      dataset = datasets.CIFAR10(self.raw_dir, train=self.train, download=True)
      images, labels = dataset.data, dataset.targets
      labels = np.array(labels).astype(np.int64)
      assert self.numSP < np.prod(np.array(dataset[0][0]).shape[:2]) and self.numSP > 1
      data_list = getDataList(images, labels, self.numSP, self.compactness, self.isShuffle, self.fullConnect)
      torch.save(self.collate(data_list), processed_path)


if __name__ == "__main__":
  root = '../data/cifar10-superpixels-128'
  dataset = CIFAR10Superpixels(root=root, train=True, numSP=128, compactness=4, shuffle=True, fullConnect=False)
  print(dataset[0])