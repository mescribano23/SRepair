import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math

class NlEncoder(nn.Module):
    def __init__(self, args):
        super(NlEncoder, self).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding(args.Vocsize, self.embedding_size)
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, self.word_len))
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.token_embedding = nn.Embedding(args.Nl_Vocsize, self.embedding_size-1)
        self.token_embedding1 = nn.Embedding(args.Nl_Vocsize, self.embedding_size)

        self.text_embedding = nn.Embedding(20, self.embedding_size)
        self.transformerBlocksTree = nn.ModuleList(
            [rightTransformerBlock(self.embedding_size, 8, self.feed_forward_hidden, 0.1) for _ in range(5)])
        self.resLinear = nn.Linear(self.embedding_size, 2)
        self.pos = PositionalEmbedding(self.embedding_size)
        self.loss = nn.CrossEntropyLoss()
        self.norm = LayerNorm(self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size // 2, int(self.embedding_size / 4), batch_first=True, bidirectional=True)
        self.conv = nn.Conv2d(self.embedding_size, self.embedding_size, (1, 10))
        self.resLinear2 = nn.Linear(self.embedding_size, 1)

    # TODO 可以检查多个bug idx loss是如何计算 能否优化
    def forward(self, input_node, inputad, res, inputtext, linenode):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(input_node, 2)
        inputad = inputad.float()
        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        x = nodeem
        lineem = self.token_embedding1(linenode)
        x = torch.cat([x, lineem], dim=1)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, inputad)
        x = x[:,:input_node.size(1)]
        resSoftmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(resmask==0, -1e9), dim=-1)
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)
        return loss, resSoftmax, x

       




class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.Tconv_forward = GCNN(dmodel=hidden)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP):
        #x = self.sublayer1(x, lambda _x: self.attention1.forward(_x, _x, _x, mask=mask))
        #x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, pos))
        #x = self.sublayer3(x, lambda _x: self.combination2.forward(_x, _x, charem))
        #print(x.size())
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        return self.dropout(x)
    
    
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            if len(list(mask.size())) != 4:
                mask = mask.unsqueeze(1).repeat(1, query.size(2), 1).unsqueeze(1)
            #print(mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn
    

class rightTransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.combination = MultiHeadedCombination(h=attn_heads, d_model=hidden)
        self.feed_forward = DenseLayer(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.conv_forward = ConvolutionLayer(dmodel=hidden, layernum=hidden)
        self.sublayer1 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer2 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer3 = SublayerConnection(size=hidden, dropout=dropout)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, charEm):
        x = self.sublayer1(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
        x = self.sublayer2(x, lambda _x: self.combination.forward(_x, _x, charEm))
        x = self.sublayer3(x, lambda _x:self.conv_forward.forward(_x, mask))
        return self.dropout(x)
    
class Embedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)
        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1, Bert=False):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size, Bert=Bert)
        self.position = PositionalEmbedding(d_model=embed_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, useline=False, lineEm = None):
        x = self.token(sequence) + self.position(sequence)
        if useline:
            x = x + lineEm
        return self.dropout(x)
    
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size=512, Bert=False):
        super(TokenEmbedding, self).__init__()
        if Bert:
            self.em = nn.Embedding(vocab_size, 768, padding_idx=0)
            self.Linear = nn.Linear(768, embed_size)
        else:
            self.em = nn.Embedding(vocab_size, embed_size, padding_idx=0)
            #super().__init__(vocab_size, embed_size, padding_idx=0)
        self.useBert = Bert
    def forward(self, inputtokens):
        out = self.em(inputtokens)
        if self.useBert:
            out = self.Linear(out)
        return out

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=None)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)
    
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    

class DenseLayer(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(DenseLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class CombinationLayer(nn.Module):
    def forward(self, query, key, value, dropout=None):
        query_key = query * key / math.sqrt(query.size(-1))
        query_value = query * value / math.sqrt(query.size(-1))
        tmpW = torch.stack([query_key, query_value], -1)
        tmpsum = torch.softmax(tmpW, dim=-1)
        tmpV = torch.stack([key, value], dim=-1)
        #print(tmpV.shape)
        #print(tmpsum.shape)
        tmpsum = tmpsum * tmpV
        tmpsum = torch.squeeze(torch.sum(tmpsum, dim=-1), -1)
        if dropout:
            tmpsum = dropout(tmpsum)
        return tmpsum
    
class ConvolutionLayer(nn.Module):
    def __init__(self, dmodel, layernum, kernelsize=3, dropout=0.1):
        super(ConvolutionLayer, self).__init__()
        self.conv1 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2)
        self.conv2 = nn.Conv1d(dmodel, layernum, kernelsize, padding=(kernelsize-1)//2)
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        mask = mask.unsqueeze(-1).repeat(1, 1, x.size(2))
        x = x.masked_fill(mask==0, 0)
        convx = self.conv1(x.permute(0, 2, 1))
        convx = self.dropout(self.activation(convx))
        out = self.conv2(convx).permute(0, 2, 1)
        return out#self.dropout(self.activation(self.conv1(self.conv2(x))))
    

class GCNN(nn.Module):
    def __init__(self, dmodel):
        super(GCNN ,self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.lstm = nn.LSTMCell(dmodel, dmodel)
    def forward(self, state, left, inputad):
        #print(state.size(), left.size())
        if left is not None:
            state = torch.cat([left, state], dim=1)
        #state = torch.cat([left, state], dim=1)
        state = self.linear(state)
        degree2 = inputad
        #degree2 = torch.sum(inputad, dim=-2, keepdim=True).clamp(min=1e-6)

        #degree = 1.0 / degree#1.0 / torch.sqrt(degree)
        #degree2 = 1.0 / torch.sqrt(degree2)
        #print(degree2.size(), state.size())
        #degree2 = degree * inputad# * degree 
        #print(degree2.size(), state.size())
        #tmp = torch.matmul(degree2, state)
        #tmp = degree2.spmm(state)
        #state = self.subconnect(state, lambda _x: _x + torch.bmm(degree2, state))
        s = state.size(1)
        #print(state.view(-1, self.hiddensize).size())
        state = self.subconnect(state, lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)) #state + torch.matmul(degree2, state)
        #state = self.subconnect(state, lambda _x: self.com(_x, _x, torch.bmm(degree2, state))) #state + torch.matmul(degree2, state)
        state = self.linearSecond(state)
        if left is not None:
            state = state[:,left.size(1):,:]
        return state#self.dropout(state)[:,50:,:]
    

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    

class MultiHeadedCombination(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.combination = CombinationLayer()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x = self.combination(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        return self.output_linear(x)