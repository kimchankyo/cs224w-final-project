import os
import shutil
import torch
import torch.nn as nn

from models import GNN
from typing import Dict, Tuple
from glob import glob
from tqdm import tqdm
from torch.optim import Optimizer
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


CHECKPOINT_NAME = 'cp_'
CHECKPOINT_EXT = 'pt'


class Trainer:
  def __init__(self, workDir: str, projectName: str, model: nn.Module, 
               optimizer: Optimizer, replace: bool = False) -> None:
    self._epoch = 0
    self._losses = []
    self._device = 'cuda' if torch.cuda.is_available() else 'cpu'

    self._modelDir = self._setModelDir(workDir, projectName, replace)
    self._model, self._opt = self._load(model, optimizer)

  def _setModelDir(self, workDir: str, name: str, replace: bool) -> str:
    workDir = os.path.abspath(workDir)
    saveDir = '{}/model'.format(workDir)
    assert os.path.exists(saveDir) and os.path.isdir(saveDir)

    modelDir = '{}/{}'.format(saveDir, name)
    if not os.path.exists(modelDir) or not os.path.isdir(modelDir): 
      os.mkdir(modelDir)
    elif replace:
      shutil.rmtree(modelDir)
      os.mkdir(modelDir)
    return modelDir

  def _load(self, model: nn.Module, opt: Optimizer) -> Tuple[nn.Module, Optimizer]:
    print('Loading Model Checkpoints at: {}'.format(self._modelDir))
    checkpoints = glob('{}/{}*.{}'.format(self._modelDir, CHECKPOINT_NAME, 
                                          CHECKPOINT_EXT))
    print('Number of Saved Checkpoints Found: {}'.format(len(checkpoints)))

    if len(checkpoints):
      checkpoints.sort()

      # Load Loss Data
      for cpFile in checkpoints:
        self._losses.append(torch.load(cpFile)['loss'])
      
      # Load Latest Training Data
      data = torch.load(checkpoints[-1])   
      self._epoch = data['epoch'] + 1

      # Load Model
      assert data['model_type'] == str(type(model))   # Ex: <class 'models.GNN'>
      modelConfig, modelState = data['model_config'], data['model_state']
      model = type(model)(**modelConfig)
      model.to(self._device)              # Hardware Acceleration
      model.load_state_dict(modelState)

      # Load Optimizer
      assert data['opt_type'] == str(type(opt))
      optState = data['opt_state']
      opt = type(opt)(model.parameters())
      opt.load_state_dict(optState)
    else:
      model.to(self._device)              # Hardware Acceleration
    return model, opt

  def _saveCheckpoint(self, loss: float) -> None:
    saveFile = '{}/{}{}.{}'.format(self._modelDir, CHECKPOINT_NAME, 
                                   self._epoch, CHECKPOINT_EXT)
    torch.save({
        'epoch': self._epoch,
        'model_type': str(type(self._model)),
        'model_config': self._model.getConfig(),    # NOTE: Custom GNN Function
        'model_state': self._model.state_dict(),
        'opt_type': str(type(self._opt)),
        'opt_state': self._opt.state_dict(),
        'loss': loss
    }, saveFile)

  def train(self, dataset: InMemoryDataset, numEpochs: int = 100, 
            trainValidSplit: float = 0.85, batchSize: int = 64, 
            lossFunc: nn.Module = nn.NLLLoss()) -> None:
    # Dataset
    trainSize = int(len(dataset)*trainValidSplit)
    trainLoader = DataLoader(dataset[:trainSize], batch_size=batchSize, 
                             shuffle=True)
    validLoader = DataLoader(dataset[trainSize:], batch_size=batchSize, 
                             shuffle=True)

    # Train Loop
    # epochsRemain = numEpochs - self._epoch  # Setting for remainind Epochs

    self._epoch -= 1
    for _ in range(numEpochs):
      self._epoch += 1
      lossIter, accIter = 0.0, 0.0
      self._model.train()
      for data in tqdm(trainLoader):
        data.to(self._device)
        self._opt.zero_grad()
        pred = self._model(data)
        label = data.y
        loss = lossFunc(pred, label)
        loss.backward()
        self._opt.step()
        lossIter += loss.item() * data.num_graphs
        accIter += torch.sum(torch.where(torch.argmax(pred, dim=1) == label, 1, 0)).cpu().numpy()
      lossIter /= len(trainLoader.dataset)
      accIter /= len(trainLoader.dataset)
      self._losses.append(lossIter)

      # Checkpoint Save
      if self._epoch % 10 == 0:
        self._saveCheckpoint(self._losses[-1])
      
      # Validation
      self._model.eval()
      accValid = 0.0
      for data in validLoader:
        with torch.no_grad():
          data.to(self._device)
          pred = self._model(data)
          label = data.y
          accValid += torch.sum(torch.where(torch.argmax(pred, dim=1) == label, 1, 0)).cpu().numpy()
      accValid /= len(validLoader.dataset)

      print("Epoch {} | Training Loss: {:.5f} | Train Acc.: {:.4f} | Valid Acc.: {:.4f}".format(self._epoch, self._losses[-1], accIter, accValid))

  def test(self, dataset: InMemoryDataset, batchSize: int = 64) -> None:
    loader = DataLoader(dataset, batch_size=batchSize)
    self._model.eval()
    acc = 0.0
    for data in loader:
      with torch.no_grad():
        data.to(self._device)
        pred = self._model(data)
        label = data.y
        acc += torch.sum(torch.where(torch.argmax(pred, dim=1) == label, 1, 0)).cpu().numpy()
    acc /= len(loader.dataset)
    print("Testing Acc.: {:.4f}".format(acc))