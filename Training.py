from matplotlib import transforms
import torch
from time import time
from torch import nn,optim as optim
from torch.utils.data import Dataset

from H5Data import h5DData

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_loader = torch.utils.data.DataLoader(h5DData("digitos.h5","train_set_x","train_set_y",transform), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(h5DData("digitos_test.h5","test_set_x","test_set_y",transform), batch_size=64, shuffle=True)


capa_entrada = 784
capas_ocultas = [128, 64]
capa_salida = 10

modelo = nn.Sequential(nn.Linear(capa_entrada, capas_ocultas[0]), nn.ReLU(),
                       nn.Linear(capas_ocultas[0], capas_ocultas[1]), nn.ReLU(),
                       nn.Linear(capas_ocultas[1], capa_salida), nn.LogSoftmax(dim=1))

j = nn.CrossEntropyLoss()

# entrenamiento de la red
optimizador = optim.Adam(modelo.parameters(), lr=0.003)
tiempo = time()
epochs = 1
for e in range(epochs):
    costo = 0
    for imagen, etiqueta in train_loader:
        
        imagen = imagen.view(imagen.shape[0], -1)
        optimizador.zero_grad()
        h = modelo(imagen.float())
        error = j(h, etiqueta.long())
        error.backward()
        optimizador.step()
        costo += error.item()
    else:
        print("Epoch {} - Funcion costo: {}".format(e, costo / len(train_loader)))
print("\nTiempo de entrenamiento (en minutes) =", (time() - tiempo) / 60)
torch.save(modelo, '/Users/gabriel1chiro/Documents/UCB/modelado/pytorch/Training.py/modelo_digitos.pt')










