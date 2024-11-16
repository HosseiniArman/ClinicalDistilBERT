import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput

class TinyBERTDistillation(nn.Module):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.attention_loss = nn.KLDivLoss(reduction="mean")
        self.hidden_loss = nn.CosineEmbeddingLoss(reduction="mean")
        self.output_loss = nn.KLDivLoss(reduction="batchmean")
        self.temperature = 1.0

    def forward(self, input_ids, attention_mask=None, labels=None, **kargs):
        student_outputs = self.student(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            output_attentions=True,
            **kargs
        )

        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                **kargs
            )

        # Extract relevant outputs
        s_attentions = student_outputs["attentions"]
        t_attentions = [att.detach() for att in teacher_outputs["attentions"]]
        s_hiddens = student_outputs["hidden_states"][1:]
        t_hiddens = [hidden.detach() for hidden in teacher_outputs["hidden_states"][1:]]
        s_logits = student_outputs["logits"]
        t_logits = teacher_outputs["logits"].detach()

        # Compute losses
        att_loss = self.compute_attention_loss(s_attentions, t_attentions, attention_mask)
        hidden_loss = self.compute_hidden_loss(s_hiddens, t_hiddens, attention_mask)
        output_loss = self.compute_output_loss(s_logits, t_logits, labels)

        total_loss = student_outputs.loss + 3.0 * (att_loss + hidden_loss) + 5.0 * output_loss

        return MaskedLMOutput(
            loss=total_loss,
            logits=student_outputs.logits,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )

    def compute_output_loss(self, s_logits, t_logits, labels):
        mask = (labels > -1).unsqueeze(-1).expand_as(s_logits).bool()
        s_logits_slct = torch.masked_select(s_logits, mask).view(-1, s_logits.size(-1))
        t_logits_slct = torch.masked_select(t_logits, mask).view(-1, s_logits.size(-1))
        output_loss = (
            self.output_loss(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return output_loss

    def compute_attention_loss(self, s_attentions, t_attentions, attention_mask):
        total_loss = None
        for s_map, t_map in zip(s_attentions, t_attentions):
            mask = attention_mask.unsqueeze(-1).expand_as(s_map).bool()
            s_map_slct = torch.masked_select(s_map, mask).view(-1, s_map.size(-1)) + 1e-12
            t_map_slct = torch.masked_select(t_map, mask).view(-1, t_map.size(-1)) + 1e-12
            att_loss = self.attention_loss(torch.log(s_map_slct), t_map_slct)
            total_loss = total_loss + att_loss if total_loss else att_loss
        return total_loss

    def compute_hidden_loss(self, s_hiddens, t_hiddens, attention_mask):
        total_loss = None
        for s_hidden, t_hidden in zip(s_hiddens, t_hiddens):
            mask = attention_mask.unsqueeze(-1).expand_as(s_hidden).bool()
            s_hidden_slct = torch.masked_select(s_hidden, mask).view(-1, s_hidden.size(-1))
            t_hidden_slct = torch.masked_select(t_hidden, mask).view(-1, s_hidden.size(-1))
            target = s_hidden.new_ones(s_hidden_slct.size(0))
            hidden_loss = self.hidden_loss(s_hidden_slct, t_hidden_slct, target)
            total_loss = total_loss + hidden_loss if total_loss else hidden_loss
        return total_loss
