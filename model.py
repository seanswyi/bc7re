import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DrugProtREModel(nn.Module):
    def __init__(self, args, config, backbone_model, tokenizer):
        super().__init__()

        self.args = args
        self.config = config

        self.bilinear_block_size = args.bilinear_block_size
        self.hidden_size = config.hidden_size
        self.max_seq_len = args.max_seq_len
        self.num_labels = args.num_labels

        self.model = backbone_model
        self.tokenizer = tokenizer

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        self.loss_function = nn.CrossEntropyLoss()

        self.head_extractor = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.tail_extractor = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
        self.bilinear = nn.Linear(in_features=(config.hidden_size * args.bilinear_block_size), out_features=args.num_labels)

    def encode(self, input_ids, attention_mask, start_tokens, end_tokens):
        _, seq_len = input_ids.shape

        start_tokens = torch.tensor(start_tokens).to(input_ids)
        end_tokens = torch.tensor(end_tokens).to(input_ids)

        start_len = start_tokens.shape[0]
        end_len = end_tokens.shape[0]

        if seq_len <= self.max_seq_len:
            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
            encoded_output = output[0]
            attention = output[-1][-1]
        else:
            new_input_ids = []
            new_attention_mask = []
            num_segments = []

            unpadded_seq_lens = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()

            for idx, seq_len_ in enumerate(unpadded_seq_lens):
                if seq_len_ <= self.max_seq_len:
                    new_input_ids.append(input_ids[idx, :self.max_seq_len])
                    new_attention_mask.append(attention_mask[idx, :self.max_seq_len])

                    num_segments.append(1)
                else:
                    input_ids1 = torch.cat([input_ids[idx, :self.max_seq_len - end_len], end_tokens], dim=-1)
                    input_ids2 = torch.cat([start_tokens, input_ids[idx, (seq_len_ - self.max_seq_len + start_len):seq_len_]], dim=-1)

                    attention_mask1 = attention_mask[idx, :self.max_seq_len]
                    attention_mask2 = attention_mask[idx, (seq_len_ - self.max_seq_len):seq_len_]

                    new_input_ids.extend([input_ids1, input_ids2])
                    new_attention_mask.extend([attention_mask1, attention_mask2])

                    num_segments.append(2)

            input_ids = torch.stack(new_input_ids, dim=0)
            attention_mask = torch.stack(new_attention_mask, dim=0)

            output = self.model(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)

            encoded_output = output[0]
            attention = output[-1][-1]

            current_idx = 0
            new_encoded_output = []
            new_attention = []

            for num_segment, seq_len_ in zip(num_segments, unpadded_seq_lens):
                if num_segment == 1:
                    output = F.pad(input=encoded_output[current_idx], pad=(0, 0, 0, seq_len - self.max_seq_len))
                    attn = F.pad(input=attention[current_idx], pad=(0, seq_len - 512, 0, seq_len - self.max_seq_len))
                    new_encoded_output.append(output)
                    new_attention.append(attn)
                elif num_segment == 2:
                    output1 = encoded_output[current_idx][:(self.max_seq_len - end_len)]
                    mask1 = attention_mask[current_idx][:(self.max_seq_len - end_len)]
                    attn1 = attention[current_idx][:, :(self.max_seq_len - end_len), :(self.max_seq_len - end_len)]

                    output1 = F.pad(input=output1, pad=(0, 0, 0, seq_len - self.max_seq_len + end_len))
                    mask1 = F.pad(input=mask1, pad=(0, seq_len - self.max_seq_len + end_len))
                    attn1 = F.pad(input=attn1, pad=(0, seq_len - self.max_seq_len + end_len, 0, seq_len - self.max_seq_len + end_len))

                    output2 = encoded_output[current_idx + 1][start_len:]
                    mask2 = attention_mask[current_idx + 1][start_len:]
                    attn2 = attention[current_idx + 1][:, start_len:, start_len:]

                    output2 = F.pad(input=output2, pad=(0, 0, seq_len_ - self.max_seq_len + start_len, seq_len - seq_len_))
                    mask2 = F.pad(input=mask2, pad=(seq_len_ - self.max_seq_len + start_len, seq_len - seq_len_))
                    attn2 = F.pad(input=attn2, pad=(seq_len_ - self.max_seq_len + start_len, seq_len - seq_len_, seq_len_ - self.max_seq_len + start_len, seq_len - seq_len_))

                    mask = mask1 + mask2 + 1e-10
                    output = (output1 + output2) / mask.unsqueeze(-1)
                    attn = attn1 + attn2
                    attn = attn / (attn.sum(-1, keepdim=True) + 1e-10)

                    new_encoded_output.append(output)
                    new_attention.append(attn)

                current_idx += num_segment

            encoded_output = torch.stack(new_encoded_output, dim=0)
            attention = torch.stack(new_attention, dim=0)

        return encoded_output, attention

    def get_representations(self, encoded_output, entity_positions, head_tail_pairs):
        head_representations = []
        tail_representations = []

        for batch_idx, head_tail_pair in enumerate(head_tail_pairs):
            entities = entity_positions[batch_idx]
            encoded_text = encoded_output[batch_idx]

            for pair in head_tail_pair:
                head_id = pair[0]
                tail_id = pair[1]

                head_entities = entities[head_id]
                tail_entities = entities[tail_id]

                head_start = head_entities[0]
                tail_start = tail_entities[0]

                head_representation = encoded_text[head_start + 1]
                tail_representation = encoded_text[tail_start + 1]

                head_representations.append(head_representation)
                tail_representations.append(tail_representation)

        return torch.stack(head_representations, dim=0), torch.stack(tail_representations, dim=0)

    def forward(self, input_ids, attention_mask, entity_positions, head_tail_pairs, labels=None):
        start_tokens = [self.cls_token_id]
        end_tokens = [self.sep_token_id]

        encoded_output, attention = self.encode(input_ids, attention_mask, start_tokens=start_tokens, end_tokens=end_tokens)

        heads, tails, label_ids = self.get_representations(encoded_output=encoded_output,
                                                           entity_positions=entity_positions,
                                                           head_tail_pairs=head_tail_pairs)

        heads_extracted = self.head_extractor(heads)
        tails_extracted = self.tail_extractor(tails)

        b1 = heads_extracted.view(-1, self.hidden_size // self.bilinear_block_size, self.bilinear_block_size)
        b2 = tails_extracted.view(-1, self.hidden_size // self.bilinear_block_size, self.bilinear_block_size)
        bl = (b1.unsqueeze(3) * b2.unsqueeze(2)).view(-1, self.hidden_size * self.bilinear_block_size)
        logits = self.bilinear(bl)

        import pdb; pdb.set_trace()

        if label_ids != []:
            label_ids = torch.tensor(label_ids)

        import pdb; pdb.set_trace()
