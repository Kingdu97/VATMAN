import math
import copy
from typing import Optional, List
import torch
from torch import nn

class ImageTransformerEncoder(nn.Module): # 이미지 특징을 인코딩하기 위한 클래스
    def __init__(self, d_model, num_layers, num_heads, dim_feedforward=2048):
        # d_model : 입력 특징 크기, n_heads : 어탠션 헤드 수, dim_ff : 피드포워드 레이어 차원수
        super(ImageTransformerEncoder, self).__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=dim_feedforward)
        self.encoder = _TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1)

    def forward(self, inputs: torch.Tensor, lens: Optional[List[int]] = None):
        # input : 이미지 특징을 나타내는 tensor
        # lens : 입력 이미지의 길이를 나타내는 list
        if lens is not None:
            max_len = max(lens)

            mask = [([False] * l + [True] * (max_len - l)) for l in lens]
            mask = torch.tensor(mask).to(device=inputs.device)
        else:
            mask = None

        inputs = inputs.permute(1, 0, 2)
        # permute를 이용하여 텐서의 차원을 변환

        inputs = inputs * math.sqrt(self.d_model)
        inputs = self.pos_encoder(inputs)

        outputs = self.encoder(src=inputs, src_key_padding_mask=mask)
        # 결과물 = (seq_len, bs, dim-2048)
        return [o.permute(1, 0, 2) for o in outputs]
        # 최종 forward 통과후 = (bs, seq_len, dim-2048)


def padTensor(t: torch.Tensor, targetLen: int) -> torch.Tensor:
    oriLen, dim = t.size()
    return torch.cat((t, torch.zeros(targetLen - oriLen, dim).to(t.device)), dim=0)
    # padTensor 함수는 주어진 2D 텐서 t를 targetLen으로 패딩합니다. 
    # 만약 t가 targetLen보다 작으면 0으로 채워진 텐서를 반환합니다.

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    # 모듈을 N번 복제한 모듈 리스트를 반환한다.

class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(_TransformerEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # 이 클래스는 인코더 레이어(encoder_layer), 레이어 수(num_layers), 정규화 모듈(norm)을 인자로 받습니다. 
        # self.layers에는 encoder_layer를 num_layers번 복제한 모듈 리스트가 저장됩니다

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        outputs = [src]

        for mod in self.layers:
            output = mod(outputs[-1], src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            outputs.append(output)

        if self.norm is not None:
            outputs[-1] = self.norm(outputs[-1])

        return outputs[1:]
        # forward 함수에서는 입력으로 src 텐서와 선택적으로 마스크(mask)와 
        # 패딩 마스크(src_key_padding_mask)를 받습니다. 
        # outputs 리스트에는 초기 입력인 src가 저장됩니다. 
        # self.layers에 있는 모든 모듈에 대해 반복하면서 출력을 계산하고, 
        # 계산된 출력을 outputs 리스트에 추가합니다. 
        # 마지막으로, norm이 None이 아니면 마지막 출력을 정규화합니다. 
        # outputs[1:]는 모든 레이어의 출력 리스트를 반환합니다.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
