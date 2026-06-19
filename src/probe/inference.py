from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe.config import ExperimentConfig
from probe.device import resolve_device, resolve_dtype


class ProbeModel:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.device = resolve_device()
        self.dtype = resolve_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model.name,
            revision=cfg.model.revision,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name,
            revision=cfg.model.revision,
            dtype=self.dtype,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers

    def get_answer_and_hidden_states(self, question: str) -> tuple[str, list]:
        prompt = self.cfg.generation.prompt_template.format(question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.cfg.generation.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_ids = generated[0, input_len:]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            forward_out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        layer_vectors = []
        for layer_idx in range(1, self.num_layers + 1):
            vec = forward_out.hidden_states[layer_idx][0, -1, :].float().cpu().numpy()
            layer_vectors.append(vec)
        return answer, layer_vectors
