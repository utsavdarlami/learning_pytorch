import torch
from torch import nn
from torchvision import models


class VGG16_Encoder(nn.Module):

    """
    Takes an image
    Returns the encodings for the image,
    """

    def __init__(self, embedding_size):

        super(VGG16_Encoder, self).__init__()

        """
            Loading the pretrained model and
            replacing the final fc to get the feature vecotor 1*1*4096
        """
        vgg16 = models.vgg16(pretrained=True)
        vgg16_features = list(vgg16.children())[:-1]
        # delete the last fc layer.
        nolast_fc = list(vgg16.classifier.children())[:-1]
        #  Done removing the final fc layer
        self.feature_vector_size = 4096
        self.feature_extractor1 = nn.Sequential(*vgg16_features)
        self.feature_extractor2 = nn.Sequential(*nolast_fc)
        self.linear = nn.Linear(self.feature_vector_size, embedding_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        About shape in forward pass
        # Image/x
          - shape : [batch_size, channel, h, w]
        # from feature extractor1
          - shape : [batch_size, 512, 7, 7]
        # x view
          - shape : [batch_size, 512*7*7]
        # from feature extractor1
          - shape : [batch_size, feature_vector_size]
        # final output
          - shape : [batch_size, embedding_size]
        """
        with torch.no_grad():
            x = self.feature_extractor1(x)
            x = x.view(x.shape[0], -1)
            x = self.feature_extractor2(x)
        x = self.relu(self.linear(x))
        return x


class RNN_Decoder(nn.Module):

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers):

        super(RNN_Decoder, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        self.embedding_layer = nn.Embedding(vocab_size, self.embedding_size)
        self.lstm_layer = nn.LSTM(self.embedding_size,
                                  self.hidden_size,
                                  self.num_layers, batch_first=True)

        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

    def forward(self, image_encodings, captions):
        """
        About shape in forward pass
        # image_encodings sh= [seq length, batch_size,embedding_size]
        # captions = [seq_length, batch_size]
        # embeding = [seq_length, batch_size, embedding_size]
        # embeding after concat = [seq_length+1, batch_size, embedding_size],
          - 1 comes from the image encondings
        # lstm_out = [seq_len+1, batch_size, hidden_size]
        # out = [seq_len+1, batch_size, vocab_size]
        """
        embeddings = self.embedding_layer(captions)
        embeddings = torch.cat((image_encodings.unsqueeze(0), embeddings),
                               dim=0)
        lstm_out, _ = self.lstm_layer(embeddings, None)
        out = self.linear(lstm_out)
        return out


class CNNtoRNN(nn.Module):

    def __init__(self, embedding_size, vocab_size, hidden_size, num_layers):

        super(CNNtoRNN, self).__init__()
        self.embedding_size = embedding_size
        self.cnnEncoder = VGG16_Encoder(self.embedding_size)
        self.rnnDecoder = RNN_Decoder(self.embedding_size,
                                      vocab_size,
                                      hidden_size,
                                      num_layers,
                                      )

    def forward(self, images, captions):
        """
        About shape in forward pass
        # images
          - shape : [batch_size, channel, h, w]
        # captions
          - shape : [seq_length, batch_size]
        # image_encodings
          - shape : [batch_size, embedding_size]
        # caption_encodings/out
          - out : [seq_len+1, batch_size, vocab_size]
        """
        images_encodings = self.cnnEncoder(images)
        captions_decodings = self.rnnDecoder(images_encodings, captions)
        return captions_decodings
