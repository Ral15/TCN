import torch
import torch.nn.functional as F
from torch import nn
from TCN.tcn import TemporalConvNet
from torch.nn import init

class TCN(nn.Module):
    def __init__(self, input_size = 1, output_size = 10, hidden_units = 25, levels = 8, kernel_size = 7, dropout = 0.25):
        super(TCN, self).__init__()
        num_channels = [hidden_units] * levels
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y1 = self.tcn(inputs)  # input should have dimension (N, C, L)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, tie_weights=False):
        super(RNNModel, self).__init__()
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity)
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight
 
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

        self.init_weights()


    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
 
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(10.)

        # orthogonal initialization of input-hidden  weights
        init.orthogonal(self.rnn.weight_ih_l0.data)

        # Identity init
        print(self.rnn.weight_hh_l0.data.shape)
        print(torch.cat([torch.eye(self.nhid) for _ in range(4)], 0))
        print(torch.cat([torch.eye(self.nhid) for _ in range(4)], 1))
        weight_hh_data = torch.cat([torch.eye(self.nhid) for _ in range(4)], 0)
        self.rnn.weight_hh_l0.data.set_(weight_hh_data)
        print(self.rnn.weight_hh_l0.data.shape)

    def forward(self, input):
        output, _ = self.rnn(input)
        decoded = self.decoder(output[-1])
        return F.log_softmax(decoded, dim=1)

    def init_hidden(self, bsz):
        #        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            c_0 = torch.randn(self.layers, bsz, self.nhid)
            h_0 = torch.randn(self.layers, bsz, self.nhid)
            return (c_0, h_0)
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)        
