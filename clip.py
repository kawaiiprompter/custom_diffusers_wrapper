import torch
import prompt_parser

class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, tokenizer, text_encoder, CLIP_stop_at_last_layers=2):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.CLIP_stop_at_last_layers = CLIP_stop_at_last_layers

    def tokenize_line(self, line):
        id_end = self.tokenizer.eos_token_id
        parsed = prompt_parser.parse_prompt_attention(line)
        tokenized = self.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]

        remade_tokens = []
        multipliers = []

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]
                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1
        
        iteration = len(remade_tokens) // 75
        rem = (75 * (iteration + 1) - len(remade_tokens))
        remade_tokens += [id_end] * rem
        multipliers += [1.0] * rem

        return remade_tokens, multipliers

    def process_text(self, texts):
        cache = {}
        remade_batch_tokens = []
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, multipliers = cache[line]
            else:
                remade_tokens, multipliers = self.tokenize_line(line)
        
                cache[line] = (remade_tokens, multipliers)
            
            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)
        return batch_multipliers, remade_batch_tokens

    def process_tokens(
        self,
        remade_batch_tokens, batch_multipliers,
        use_old_emphasis_implementation=False,
        ):
        self.device = self.text_encoder.device
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
        if not use_old_emphasis_implementation:
            remade_batch_tokens = [[id_start] + x[:75] + [id_end] for x in remade_batch_tokens]
            batch_multipliers = [[1.0] + x[:75] + [1.0] for x in batch_multipliers]

        tokens = torch.asarray(remade_batch_tokens).to(self.device)
        outputs = self.text_encoder(input_ids=tokens, output_hidden_states=-self.CLIP_stop_at_last_layers)

        if self.CLIP_stop_at_last_layers > 1:
            z = outputs.hidden_states[-self.CLIP_stop_at_last_layers]
            z = self.text_encoder.text_model.final_layer_norm(z)
        else:
            z = outputs.last_hidden_state

        # restoring original mean is likely not correct, but it seems to work well to prevent artifacts that happen otherwise
        batch_multipliers_of_same_length = [x + [1.0] * (75 - len(x)) for x in batch_multipliers]
        batch_multipliers = torch.asarray(batch_multipliers_of_same_length).to(self.device)
        original_mean = z.mean()
        z *= batch_multipliers.reshape(batch_multipliers.shape + (1,)).expand(z.shape)
        new_mean = z.mean()
        z *= original_mean / new_mean

        return z

    def forward(self, text):
        batch_multipliers, remade_batch_tokens = self.process_text(text)
        z = self.process_tokens(remade_batch_tokens, batch_multipliers)
        return z
