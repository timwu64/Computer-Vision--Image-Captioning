import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)

        # the linear layer that maps the hidden state output dimension 
        # to the number of vocab we want as output, vocab_size
        self.hidden2vocab = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        ''' Define the feedforward behavior of the model.'''
        # create embedded word vectors for each word in a sentence
        embeds = self.word_embeddings(captions[:,:-1]) # apply [:-1] to ignore the last character <end>
        
        inputs = torch.cat((features.unsqueeze(dim=1),embeds), dim=1)
        
        # get the output by passing the lstm over our word embeddings
        # the lstm takes in our embeddings
        lstm_out, _ = self.lstm(inputs)
        
        # get the scores for the most likely tag for a word
        outputs = self.hidden2vocab(lstm_out)
        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            tag_outputs = self.hidden2vocab(lstm_out.squeeze(1))
            predicted = tag_outputs.argmax(dim=1)
            result.append(predicted.item())
            inputs = self.word_embeddings(predicted).unsqueeze(dim=1)
        return result