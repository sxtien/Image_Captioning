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
        super(DecoderRNN,self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,vocab_size,batch_first=True)
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.fc = nn.Linear(hidden_size,vocab_size)
    
    def forward(self, features, captions):
        # encoderCNN output features
        features = features.view(features.size()[0],1,-1)
        # print(features.size())
        # training captions, take the first len-1 elements
        captions = self.embed(captions[:,:-1])
        # print(captions.size())
        concat_features = torch.cat((features,captions),dim=1)
        # print(concat_features.size())
        outputs,_ = self.lstm(concat_features)
        outputs = self.fc(outputs)
        
        return outputs

       

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        result = []
        for i in range(max_len):
            outputs,states = self.lstm(inputs,states)
            #print("lstm outouts size", outputs.size())
            outputs = self.fc(outputs)
            #print("fc outputs size: ",outputs.size())
            vocab_idx = outputs.max(2)[1]
            #print("vocabulary index: ",vocab_idx)
            result.append(vocab_idx.item())
            # vocob_idx size is 1*1,dont need to unqueeze
            inputs = self.embed(vocab_idx)
            #print("inputs embed size:", inputs.size())
            #break
        return result


        
    