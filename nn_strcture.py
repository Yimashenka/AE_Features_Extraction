import SAE
import torch
import numpy as np

from torch import nn, optim
from torch.autograd import Variable

from preprocessing_data import nb_users, training_set, nb_movies, test_set

sae = SAE.SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(
    sae.parameters(),
    lr=0.01,
    weight_decay=0.5
)

# Training the structure
nb_epoch = 200

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies/float(torch.sum(target.data > 0 ) +1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1.
            optimizer.step()
    print(f'epoch: {epoch}     loss: {train_loss/s}')

# Testing the SAE
test_loss = 0
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies/float(torch.sum(target.data > 0 ) +1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print(f'test loss: {test_loss/s}')