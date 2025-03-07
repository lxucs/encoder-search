import torch
import torch.nn as nn
import logging
from os.path import join, exists
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class ColbertModel(nn.Module):

    def __init__(self, encoder_path: str=None, use_linear=False, linear_dim=None,
                 normalize=True, pooling_type: str='cls'):
        super().__init__()
        self.normalize = normalize
        self.pooling_type = pooling_type
        assert self.pooling_type in ('cls', 'mean')

        self.encoder = AutoModel.from_pretrained(encoder_path)
        self.colbert_linear = torch.nn.Linear(in_features=self.encoder.config.hidden_size,
                                              out_features=linear_dim or self.encoder.config.hidden_size) if use_linear else None
        colbert_path = join(encoder_path, 'colbert_linear.pt')
        if use_linear:
            if exists(colbert_path):
                self.colbert_linear.load_state_dict(torch.load(colbert_path, map_location='cpu', weights_only=True))
                print(f'Loaded colbert linear')
            else:
                print('Did not find colbert linear; dismiss')

    @property
    def device(self):
        return self.encoder.device

    def save_weights(self, save_dir):
        self.encoder.save_pretrained(save_dir)
        if self.colbert_linear:
            torch.save(self.colbert_linear.state_dict(), f'{save_dir}/colbert_linear.pt')

    def encode(self, batch, remove_colbert_padding=True):
        # Encode
        input_ids, attention_mask = batch['input_ids'], batch['attention_mask']
        batch = {'input_ids': input_ids, 'attention_mask': attention_mask,
                  'output_attentions': False, 'output_hidden_states': False, 'return_dict': True}
        last_hidden_state = self.encoder(**batch)['last_hidden_state']  # [bsz, seq_len, hidden]

        # Important: set padding hidden to 0
        last_hidden_state *= attention_mask.unsqueeze(-1).float()

        # Pooled hidden
        if self.pooling_type == 'cls':
            pooled_hidden = last_hidden_state[:, 0]
        else:
            pooled_hidden = last_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

        # Colbert hidden
        if self.colbert_linear:
            colbert_hidden = self.colbert_linear(last_hidden_state[:, 1:])  # [bsz, seq_len-1, hidden]
        else:
            colbert_hidden = last_hidden_state[:, 1:]

        # Normalize in the end
        if self.normalize:
            pooled_hidden = nn.functional.normalize(pooled_hidden, p=2, dim=-1)
            colbert_hidden = nn.functional.normalize(colbert_hidden, p=2, dim=-1)

        if remove_colbert_padding:
            colbert_hidden = [row_hidden[: row_mask.sum() - 1]  # [seq_len_wo_pad, hidden] for each
                              for row_hidden, row_mask in zip(colbert_hidden, attention_mask)]
        return pooled_hidden, colbert_hidden

    @classmethod
    def get_batch_colbert_sim(cls, q_hidden, c_hidden, q_mask):
        token_sim = torch.einsum('qin,cjn->qicj', q_hidden, c_hidden)  # qicj
        sim, _ = token_sim.max(dim=-1)  # qic
        sim = sim.sum(1) / q_mask[:, 1:].sum(-1, keepdim=True)  # qc
        return sim

    @classmethod
    def get_batch_colbert_sim_w_simple_query(cls, q_hidden, c_hidden):
        token_sim = torch.einsum('qn,cjn->qcj', q_hidden, c_hidden)  # qcj
        sim, _ = token_sim.max(dim=-1)  # qc
        return sim
