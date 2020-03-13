import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as op
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets


class HateSpeechDetector(nn.Module):
    def __init__(self, device, vocabularySize, output, embedding, hidden, layers, dropProb=0.5):
        super(HateSpeechDetector, self).__init__()
        #Number of outputs (Classes/Categories)
        self.output = output
        #Number of layers in the LSTM
        self.numLayers = layers
        #Number of hidden neurons in each LSTM layer
        self.hiddenDimensions = hidden
        #Device being used for by model (CPU or GPU)
        self.device = device
        
        #Embedding layer finds correlations in words by converting word integers into vectors
        self.embedding = nn.Embedding(vocabularySize, embedding)
        #LSTM stores important data in memory, using it to help with future predictions
        self.lstm = nn.LSTM(embedding,hidden,layers,dropout=dropProb,batch_first=True)
        #Dropout is used to randomly drop nodes. This helps to prevent overfitting of the model during training
        self.dropout = nn.Dropout(dropProb)

        #Establishing 4 simple layers and a sigmoid output
        self.fc = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, output)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden):
        batchSize = x.size(0)
        print(batchSize)
        x = x.long()
        print(x)
        embeds = self.embedding(x)
        print(embeds.shape)
        lstm_out, hidden = self.lstm(embeds, hidden)
        print(lstm_out.shape)
        lstm_out = lstm_out.contiguous().view(-1,self.hiddenDimensions)
        print(lstm_out.shape)
        out = self.dropout(lstm_out)
        out = self.fc(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)
        out = self.fc5(out)
        out = self.fc6(out)
        out = self.softmax(out)
        print(out.shape)
        out = out.view(batchSize, 3)
        return out, hidden

    def init_hidden(self, batchSize, device):
        weight = next(self.parameters()).data
        print(weight)
        hidden = (weight.new(self.numLayers, batchSize, self.hiddenDimensions).zero_().to(device), weight.new(self.numLayers, batchSize, self.hiddenDimensions).zero_().to(device))
        print(hidden)
        return hidden