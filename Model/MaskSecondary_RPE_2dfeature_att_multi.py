import torch
import torch.nn as nn
import copy
import math

class Intermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.act = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.act(hidden_states)

        return hidden_states


class Output(nn.Module):
    def __init__(self, intermediate_size, embed_size, hidden_dropout_prob):
        super(Output, self).__init__()
        self.dense = nn.Linear(intermediate_size, embed_size)
        self.LayerNorm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor): 
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.pa(x)
        return x * y


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Block2d(nn.Module):

    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, dropout, dilation=1):
        super(Block2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding ,bias=True, dilation=dilation)

        self.act1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=padding, bias=True, dilation=dilation)

        self.calayer = CALayer(out_channel)

        self.palayer = PALayer(out_channel)


    def forward(self, x):
        res = self.conv1(x)
        res = self.act1(res)
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x

        return res


def scaled_dot_product_attention(q, k, v, position_bias, mask, rp_mask):

    qk = torch.matmul(q, k.transpose(-1, -2))
    depth = k.size(-1)
    logits = (qk + position_bias) / math.sqrt(depth)

    if mask is not None:

        logits += (mask * -1e9)
        logits += (rp_mask * -1e9)

    attention = logits
    attention_weights = nn.Softmax(dim=-1)(logits)
    context = torch.matmul(attention_weights, v)

    return attention, context


class MultiHeadAttention(nn.Module):
    def __init__(self,embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

        assert embed_size % self.num_heads == 0

        self.depth = embed_size // self.num_heads

        self.q_dense = nn.Linear(embed_size, embed_size)
        self.k_dense = nn.Linear(embed_size, embed_size)
        self.v_dense = nn.Linear(embed_size, embed_size)

        self.dense = nn.Linear(embed_size, embed_size)

        self.has_relative_attention_bias = True

        self.relative_attention_num_buckets = 16
        self.max_distance = 16

        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)

    def _relative_position_bucket(self, relative_position, bidirectional=False, num_buckets=16, max_distance=32):

        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))

        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        relative_postion_if_large = max_exact + (
                torch.log(relative_position.float() / max_exact)
                / math.log(self.max_distance / max_exact)
                * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        context_position = torch.arange(
            query_length, dtype=torch.long, device='cuda'
        )[:, None]
        memory_position = torch.arange(
            key_length, dtype=torch.long, device='cuda'
        )[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = self._relative_position_bucket(relative_position)
        relative_position_bucket = relative_position_bucket.transpose(-1, -2) + relative_position_bucket
        RP_mask = relative_position_bucket == (self.max_distance - 1)
        values = self.relative_attention_bias(relative_position_bucket)
        values = values.permute([2, 0, 1]).unsqueeze(0)
        return values, RP_mask.cuda()

    def split_heads(self, inputs, batch_size):
        inputs = inputs.reshape(batch_size, -1, self.num_heads, self.depth)
        return inputs.permute(0, 2, 1, 3)

    def forward(self, query, key, value, mask):
        batch_size = query.size(0)

        q = self.q_dense(query)
        k = self.k_dense(key)
        v = self.v_dense(value)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        position_bias, rp_mask = self.compute_bias(q.size(-2), k.size(-2))
        attention, scaled_attention = scaled_dot_product_attention(q, k, v, position_bias, mask, rp_mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        concat_attention = scaled_attention.reshape(batch_size, -1, self.embed_size)

        outputs = self.dense(concat_attention)

        return outputs, attention


class Encoder_layer(nn.Module):
    def __init__(self, embed_size, hidden_size, num_heads, dropout):
        super(Encoder_layer, self).__init__()

        self.attention = MultiHeadAttention(embed_size,num_heads)
        self.intermediate = Intermediate(embed_size, hidden_size)
        self.output = Output(hidden_size, embed_size, dropout)
        self.LayerNorm = nn.LayerNorm(embed_size, eps=1e-6)
        self.dropout=nn.Dropout(dropout)

    def forward(self, x, mask):

        attention, real_attention = self.attention(x, x, x, mask)
        attention = self.dropout(attention)
        attention = self.LayerNorm(x+attention)
        intermediate_output = self.intermediate(attention)
        layer_output = self.output(intermediate_output, attention)

        return layer_output, real_attention


class Encoder_MultipleLayers(nn.Module):
    def __init__(self, num_layers, embed_size, hidden_size, num_heads, dropout):
        super(Encoder_MultipleLayers, self).__init__()
        layer = Encoder_layer(embed_size, hidden_size, num_heads, dropout)
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])

    def forward(self, h, mask):
        atts = []
        for layer in self.layers:
            h, att=layer(h, mask)
            atts.append(att)

        return h, atts

def symmetrize(x):
    return (x + x.transpose(-1, -2))/2


class MASKSecondary(nn.Module):
    def __init__(self, dropout, dssp_dim):
        super(MASKSecondary, self).__init__()
        esm_dim = 1024  

        self.relu = nn.ReLU()
        self.elu = nn.ELU()
        seg_dropout = 0.0
        self.dropout = nn.Dropout(dropout)

        self.num_layers = 3

        self.num_heads = 8

        self.encoder = Encoder_MultipleLayers(self.num_layers, esm_dim, 1024, self.num_heads,
                                              seg_dropout)

        input_dim = self.num_heads * self.num_layers

        self.first = nn.Conv2d(input_dim, 64, 5, 1, 2)

        self.segment_detector_1 = nn.ModuleList([Block2d(64, 64, 5, 1, 2, dropout, dilation=1),
                                                 Block2d(64, 64, 5, 1, 2, dropout, dilation=1),
                                                 Block2d(64, 64, 5, 1, 2, dropout, dilation=1)
                                                 ])

        self.segment_detector_2 = nn.ModuleList([Block2d(64, 64, 3, 1, 2, dropout, dilation=2),
                                                 Block2d(64, 64, 3, 1, 2, dropout, dilation=2),
                                                 Block2d(64, 64, 3, 1, 2, dropout, dilation=2)
                                                 ])

        self.segment_detector_3 = nn.ModuleList([Block2d(64, 64, 3, 1, 1, dropout, dilation=1),
                                                 Block2d(64, 64, 3, 1, 1, dropout, dilation=1),
                                                 Block2d(64, 64, 3, 1, 1, dropout, dilation=1)
                                                 ])


        dim = 64
        out_dim = dim*3
        self.fc1 = nn.Linear(out_dim, out_dim // 2)
        self.fc2 = nn.Linear(out_dim // 2, dssp_dim)



        self.tanh = nn.Tanh()


    def forward(self, x, att, mask):

        _, attention = self.encoder(x, mask)

        attention = torch.cat(attention, dim=1)
        attention = symmetrize(attention)

        attention = h = self.tanh(attention)

        h1=h2=h3=self.first(h)
        for i in range(len(self.segment_detector_3)):
            h1 = self.segment_detector_1[i](h1)
            h2 = self.segment_detector_2[i](h2)
            h3 = self.segment_detector_3[i](h3)


        features = torch.cat([h1, h2, h3], dim=1)
        diag = torch.diagonal(features, dim1=-2, dim2=-1).transpose(-1, -2)
        out = self.dropout(self.relu(self.fc1(diag)))
        out = self.fc2(out)

        return (diag, out, None), attention
