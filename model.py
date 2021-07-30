import logging
import sys

import numpy as np
from opt_einsum import contract
import torch
import torch.nn as nn
import torch.nn.functional as F

from at_loss import ATLoss


logger = logging.getLogger()


class DrugProtREModel(nn.Module):
    def __init__(self, args, config, backbone_model, tokenizer):
        super().__init__()

        self.args = args
        self.config = config
        self.setting = args.setting

        self.adaptive_thresholding_k = args.adaptive_thresholding_k
        self.bilinear_block_size = args.bilinear_block_size
        self.classification_type = args.classification_type
        self.classifier_type = args.classifier_type
        self.hidden_size = config.hidden_size
        self.max_seq_len = args.max_seq_len
        self.num_labels = args.num_labels
        self.use_at_loss = args.use_at_loss
        self.use_attention = args.use_attention

        self.model = backbone_model
        self.tokenizer = tokenizer

        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id

        if args.use_at_loss:
            self.loss_function = ATLoss()
        elif not args.use_at_loss:
            self.loss_function = nn.CrossEntropyLoss()

        if args.use_attention:
            self.head_extractor = nn.Linear(in_features=(2 * config.hidden_size), out_features=config.hidden_size)
            self.tail_extractor = nn.Linear(in_features=(2 * config.hidden_size), out_features=config.hidden_size)
            self.cls_extractor = nn.Linear(in_features=(2 * config.hidden_size), out_features=config.hidden_size)
        elif not args.use_attention:
            self.head_extractor = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
            self.tail_extractor = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)
            self.cls_extractor = nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size)

        if (args.classifier_type == 'bilinear') and (args.classification_type == 'both'):
            self.classifier = nn.Linear(in_features=(config.hidden_size * args.bilinear_block_size * 2), out_features=args.num_labels)
        elif (args.classifier_type == 'bilinear') and (args.classification_type == 'entity_marker'):
            self.classifier = nn.Linear(in_features=(config.hidden_size * args.bilinear_block_size), out_features=args.num_labels)
        elif (args.classifier_type == 'linear') and (args.classification_type == 'cls'):
            self.classifier = nn.Linear(in_features=config.hidden_size, out_features=args.num_labels)
        elif (args.classifier_type == 'linear') and (args.classification_type == 'entity_marker'):
            self.classifier = nn.Linear(in_features=(config.hidden_size * 2), out_features=args.num_labels)

        if args.use_attention and (args.classifier_type == 'linear'):
            self.classifier = nn.Linear(in_features=(2 * config.hidden_size), out_features=args.num_labels)

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

    def get_representations(self, encoded_output, attention, entity_positions, head_tail_pairs, setting='document'):
        head_representations = []
        tail_representations = []
        attention_representations = []
        cls_representations = []

        import pdb; pdb.set_trace()

        for batch_idx, head_tail_pair in enumerate(head_tail_pairs):
            entities = entity_positions[batch_idx]
            encoded_text = encoded_output[batch_idx]
            batch_attention = attention[batch_idx]

            if setting == 'document':
                cls_representation = encoded_text[0]
                cls_representation_tiled = cls_representation.repeat(repeats=(len(head_tail_pair), 1))
                cls_representations.append(cls_representation_tiled)

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

                    head_attention = batch_attention[:, head_start + 1]
                    tail_attention = batch_attention[:, tail_start + 1]
                    head_tail_attention = (head_attention * tail_attention).mean(0)
                    attention_representation = contract('rl, ld -> rd', head_tail_attention.unsqueeze(0), encoded_text)

                    attention_representations.append(attention_representation)
            elif setting == 'sentence':
                cls_representation = encoded_text[0]
                cls_representation_tiled = cls_representation.repeat(repeats=(1, 1))
                cls_representations.append(cls_representation_tiled)

                head_id = head_tail_pair[0]
                tail_id = head_tail_pair[1]

                head_entities = entities[head_id]
                tail_entities = entities[tail_id]

                head_start = head_entities[0]
                tail_start = tail_entities[0]

                head_representation = encoded_text[head_start + 1]
                tail_representation = encoded_text[tail_start + 1]

                head_representations.append(head_representation)
                tail_representations.append(tail_representation)

                head_attention = batch_attention[:, head_start + 1]
                tail_attention = batch_attention[:, tail_start + 1]
                head_tail_attention = (head_attention * tail_attention).mean(0)
                attention_representation = contract('rl, ld -> rd', head_tail_attention.unsqueeze(0), encoded_text)

                attention_representations.append(attention_representation)

        import pdb; pdb.set_trace()
        try:
            head_representations = torch.stack(head_representations, dim=0)
            tail_representations = torch.stack(tail_representations, dim=0)
            attention_representations = torch.cat(attention_representations, dim=0)
            cls_representations = torch.cat(cls_representations, dim=0)
        except RuntimeError as e:
            logger.error(e)
            logger.error(f"head_representations = {head_representations}\ntail_representations = {tail_representations}")
            sys.exit()

        return head_representations, tail_representations, attention_representations, cls_representations

    def get_predictions(self, logits, sentence_ids, use_at_loss=True, k=-1):
        if use_at_loss:
            threshold_logit = logits[:, 0].unsqueeze(1)
            predictions = torch.zeros_like(logits).to(logits)
            mask = logits > threshold_logit

            if k > 0:
                top_v, _ = torch.topk(logits, k, dim=1)
                top_v = top_v[:, -1]
                mask = (logits >= top_v.unsqueeze(1)) & mask

            predictions[mask] = 1.0
            predictions[:, 0] = (predictions.sum(1) == 0.).to(logits)
        else:
            predictions = torch.argmax(logits, dim=-1)

        # If the entity pair is inter-sentence, then no-relation is predicted.
        # for idx, id_pair in enumerate(sentence_ids):
        #     head_sentence_id = id_pair[0]
        #     tail_sentence_id = id_pair[1]

        #     if head_sentence_id != tail_sentence_id:
        #         if use_at_loss:
        #             true_idx = torch.argmax(predictions[idx])
        #             predictions[idx][true_idx] = 0
        #             predictions[idx][0] = 1
        #         else:
        #             predictions[idx] = 0

        return predictions

    def forward(self, input_ids, attention_mask, entity_positions, head_tail_pairs, sentence_ids, labels=None):
        start_tokens = [self.cls_token_id]
        end_tokens = [self.sep_token_id]

        encoded_output, attention = self.encode(input_ids, attention_mask, start_tokens=start_tokens, end_tokens=end_tokens)
        heads, tails, attentions, clss = self.get_representations(encoded_output=encoded_output,
                                                                  attention=attention,
                                                                  entity_positions=entity_positions,
                                                                  head_tail_pairs=head_tail_pairs,
                                                                  setting=self.setting)

        if self.classification_type == 'cls':
            representations = clss
        elif self.classification_type in ['both', 'entity_marker']:
            if self.use_attention:
                heads = torch.cat([heads, attentions], dim=-1)
                tails = torch.cat([tails, attentions], dim=-1)
                clss = torch.cat([clss, attentions], dim=-1)

            if self.classifier_type == 'linear':
                if self.classification_type == 'both':
                    representations = torch.cat([clss, heads, tails], dim=-1)
                elif self.clasification_type == 'entity_marker':
                    representations = torch.cat([heads, tails], dim=-1)
            elif self.classifier_type == 'bilinear':
                heads_extracted = self.head_extractor(heads)
                tails_extracted = self.tail_extractor(tails)

                if self.classification_type == 'both':
                    cls_extracted = self.cls_extractor(clss)
                    heads_extracted = torch.cat([heads_extracted, cls_extracted], dim=-1)
                    tails_extracted = torch.cat([tails_extracted, cls_extracted], dim=-1)

                    temp1 = heads_extracted.view(-1, self.hidden_size // self.bilinear_block_size * 2, self.bilinear_block_size)
                    temp2 = tails_extracted.view(-1, self.hidden_size // self.bilinear_block_size * 2, self.bilinear_block_size)
                    representations = (temp1.unsqueeze(3) * temp2.unsqueeze(2)).view(-1, self.hidden_size * self.bilinear_block_size * 2)
                elif self.classification_type == 'entity_marker':
                    temp1 = heads_extracted.view(-1, self.hidden_size // self.bilinear_block_size, self.bilinear_block_size)
                    temp2 = tails_extracted.view(-1, self.hidden_size // self.bilinear_block_size, self.bilinear_block_size)
                    representations = (temp1.unsqueeze(3) * temp2.unsqueeze(2)).view(-1, self.hidden_size * self.bilinear_block_size)

        logits = self.classifier(representations)

        sentence_ids = sum(sentence_ids, [])
        assert len(sentence_ids) == logits.shape[0], f"len(sentence_ids) = {len(sentence_ids)} and logits.shape[0] = {logits.shape[0]}"

        predictions = self.get_predictions(logits=logits, sentence_ids=sentence_ids, use_at_loss=self.use_at_loss, k=self.adaptive_thresholding_k)

        output = (predictions,)

        if labels:
            if self.use_at_loss:
                labels = [torch.tensor(label) for label in labels]
                labels = torch.cat(labels, dim=0).to(logits).float()
            elif not self.use_at_loss:
                labels = [torch.tensor(label) for label in sum(labels, [])]
                labels = torch.tensor([torch.argmax(label) for label in labels]).to(logits).long()

            loss = self.loss_function(logits.float(), labels)

            output += (loss,)

        return output
