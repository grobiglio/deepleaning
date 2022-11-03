import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from gensim import corpora
import pandas as pd
from tqdm import tqdm
from practico1_modulo import *
from clean_console import clean_console

ARCHIVO_SET_DE_ENTRENAMIENTO = './data/training_set.csv'
ARCHIVO_SET_DE_ENTRENAMIENTO_REDUCIDO = './data/training_set_reduced.csv'
ARCHIVO_SET_DE_PRUEBA = './data/test_set.csv'
ARCHIVO_SET_DE_VALIDACION = './data/validation_set.csv'
ARCHIVO_DE_EMBEDDINGS = 'D:/SBW-vectors-300-min5.txt'
ARCHIVO_DICCIONARIO = './data/diccionario.txt'
EPOCHS = 5
VOCAB_SIZE = 1000 # Cantidad de palabras en el diccionario
MUESTRAS = 10000 # Cantidad de muestras que se toman del set de entrenamiento
                  # Si se configura valor 0 (cero) se toman todas las muestras
TOKENS_ESPECIALES = {'[relleno]': 0, '[desconocido]': 1}
VALOR_DE_RELLENO = 0
BATCH_SIZE = 100

clean_console()

# Carga de datos
print('Cargando datos...')
df_entrenamiento = pd.read_csv(ARCHIVO_SET_DE_ENTRENAMIENTO_REDUCIDO)
if MUESTRAS > 0:
    titulos = df_entrenamiento.sample(MUESTRAS).title.to_list()
else:
    titulos = df_entrenamiento.title.to_list()
print('Datos cargados con éxito, a continuación una muestra de los datos.')
print(df_entrenamiento.head())
print('-'*50+'\n')

# Procesamiento de los títulos
corpus_titulos = []
for titulo in tqdm(titulos, desc="Procesando títulos"):
    titulo_procesado = procesar_titulo(titulo)
    corpus_titulos.append(titulo_procesado)
# corpus_titulos contiene todos los títulos procesados.
# Cada título es una lista de palabras procesadas
print(f'Se han procesado {len(corpus_titulos)} títulos.\nMuestra de los 5 primeros:')
for i in range(5):
    print(corpus_titulos[i])
print('-'*50+'\n')

# Construcción de un diccionario a partir del corpus de títulos
# https://radimrehurek.com/gensim/corpora/dictionary.html
print('Generando diccionario...')
diccionario = corpora.Dictionary(corpus_titulos)
diccionario.filter_extremes(no_below=2, no_above=0.5, keep_n=VOCAB_SIZE)
diccionario.patch_with_special_tokens(TOKENS_ESPECIALES)
diccionario.compactify()
print(f'Se generó un diccionario de longitud {len(diccionario)} a partir de {diccionario.num_docs} documentos.')
print('-'*50+'\n')

# Guardado del diccionario en archivo de texto
diccionario.save_as_text(ARCHIVO_DICCIONARIO, sort_by_word=True)
print(f'Se guardó el diccionario en {ARCHIVO_DICCIONARIO}.')
print('-'*50+'\n')

# Encoding de datos
encoded_titulos = []
for titulo in tqdm(corpus_titulos, desc='Encoding de títulos'):
    encoded_titulo = diccionario.doc2idx(titulo, unknown_word_index=1)
    encoded_titulos.append(encoded_titulo)
print('Se realizó el encoding de los títulos.\nMuestra de los 5 primeros:')
for i in range(5):
    print(encoded_titulos[i])
print('-'*50+'\n')

# Completamiento de datos
print('Completamiento de datos.')
longitudes_titulos = [len(titulo) for titulo in encoded_titulos]
longitud_maxima = max(longitudes_titulos)
print(f'El título más largo tiene {longitud_maxima} palabras/indices.')
print(f'Se rellenará con {VALOR_DE_RELLENO} los valores faltantes en los títulos que tengan menor longitud.')
data = [d[:ele] + [VALOR_DE_RELLENO] * (longitud_maxima - ele) for d, ele in zip(encoded_titulos, longitudes_titulos)]
for i in range(5):
    print(data[i])
print('-'*50+'\n')
X = torch.LongTensor(data)

# Conversión de categorías a etiquetas
idx_to_target = sorted(df_entrenamiento["category"].unique())
target_to_idx = {t: i for i, t in enumerate(idx_to_target)}
def encode_target(target):
    # Convierte las categorías a etiquetas
    return target_to_idx[target]
categoria_etiquetada = [encode_target(t) for t in df_entrenamiento['category']]
print('Se etiquetaron las categorías.\nMuestra de las 5 primeras:')
for i in range(5):
    print(categoria_etiquetada[i])
print('-'*50+'\n')
y = torch.LongTensor(categoria_etiquetada)

torch.save(X, './data/X_train.pt')
torch.save(y, './data/y_train.pt')

# Embedding de títulos
# https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
print('Embeddings.')
vector_size = 300
embeddings_matrix = torch.randn(len(diccionario), vector_size)
embeddings_matrix[0] = torch.zeros(vector_size)
print(f'Tamaño de la matriz de embeddings: {embeddings_matrix.shape}.')
with open(ARCHIVO_DE_EMBEDDINGS, encoding='utf-8', mode='r') as file:
    for line in tqdm(file, total=1000654, desc="Recorriendo archivo de embeddings"):
        word, vector = line.strip().split(None, 1)
        if word in diccionario.token2id:
            embeddings_matrix[diccionario.token2id[word]] = torch.FloatTensor([float(n) for n in vector.split()])
embeddings = nn.Embedding.from_pretrained(embeddings_matrix,
                                          padding_idx=0)
print(f'Finalizado el embedding de tamaño {embeddings.weight.shape}.')
print('-'*50+'\n')

train_dataset = MeLiChallengeDataset(X, y)
print(len(train_dataset))
print(train_dataset[10])
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=False)
i = 0
for data in tqdm(train_loader):
    i += 1
print(f'{i} iteraciones')