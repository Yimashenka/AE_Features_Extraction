from torch import nn
from preprocessing_data import nb_movies
class SAE(nn.Module):
    def __init__(self):
        super(SAE, self).__init__()

        # fc means Full Connection
        self.fc1 = nn.Linear(nb_movies, 20) #20 = 20 neurons
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(
            self.fc1(x)
        )
        x = self.activation(
            self.fc2(x)
        )
        x = self.activation(
            self.fc3(x)
        )
        x = self.fc4(x)
        return x