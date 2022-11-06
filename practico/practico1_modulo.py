from gensim.parsing import preprocessing
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class MeLiChallengeDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        item = {
            "data": self.X[item],
            "target": self.y[item]
        }
        return item


class MeLiChallengeClassifier(nn.Module):
    def __init__(self, embeddings):
        super().__init__()
        self.vector_size = embeddings.weight.shape[1]
        self.embeddings = embeddings
        self.hidden1 = nn.Linear(self.vector_size, 300)
        self.hidden2 = nn.Linear(300, 500)
        self.output = nn.Linear(500, 632)
    
    def forward(self, x):
        x = self.embeddings(x)
        x = torch.mean(x, dim=1)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = torch.sigmoid(self.output(x))
        return x


def procesar_titulo(titulo: str) -> list:
    '''Procesa una oración o cadena de titulo.
    Parámetro:
    '''
    titulo_procesado_1 = titulo.lower()
    titulo_procesado_2 = preprocessing.split_alphanum(titulo_procesado_1)
    titulo_procesado_3 = preprocessing.strip_punctuation(titulo_procesado_2)
    titulo_procesado_4 = preprocessing.strip_numeric(titulo_procesado_3)
    titulo_procesado_5 = preprocessing.strip_multiple_whitespaces(titulo_procesado_4)
    titulo_procesado_6 = preprocessing.strip_short(titulo_procesado_5, minsize=4)
    titulo_procesado_7 = preprocessing.remove_stopwords(titulo_procesado_6, stopwords=stop_words)
    titulo_procesado = titulo_procesado_7.split()
    return titulo_procesado


archivo_stopwords_spanish = './data/spanish_stopwords.txt'
stop_words = []
with open(archivo_stopwords_spanish, encoding='utf-8') as file:
    for line in file:
        stop_words.append(line.strip())