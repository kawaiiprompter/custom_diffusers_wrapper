import math
import torch
import prompt_parser

def get_target_prompt_token_count(token_count):
    return math.ceil(max(token_count, 1) / 75) * 75

class FrozenCLIPEmbedderWithCustomWords(torch.nn.Module):
    def __init__(self, tokenizer, text_encoder,
                 CLIP_stop_at_last_layers=2,
                 enable_emphasis=True,
                 comma_padding_backtrack=20
                ):
        super().__init__()
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.CLIP_stop_at_last_layers = CLIP_stop_at_last_layers
        self.enable_emphasis = enable_emphasis
        self.comma_padding_backtrack = comma_padding_backtrack
        
        self.comma_token = [v for k, v in self.tokenizer.get_vocab().items() if k == ',</w>'][0]

    def tokenize_line(self, line, used_custom_terms):
        id_end = self.tokenizer.eos_token_id

        if self.enable_emphasis:
            parsed = prompt_parser.parse_prompt_attention(line)
        else:
            parsed = [[line, 1.0]]

        tokenized = self.tokenizer([text for text, _ in parsed], truncation=False, add_special_tokens=False)["input_ids"]
        
        remade_tokens = []
        multipliers = []
        last_comma = -1

        for tokens, (text, weight) in zip(tokenized, parsed):
            i = 0
            while i < len(tokens):
                token = tokens[i]

                if token == self.comma_token:
                    last_comma = len(remade_tokens)
                elif self.comma_padding_backtrack != 0 and max(len(remade_tokens), 1) % 75 == 0 and last_comma != -1 and len(remade_tokens) - last_comma <= self.comma_padding_backtrack:
                    last_comma += 1
                    reloc_tokens = remade_tokens[last_comma:]
                    reloc_mults = multipliers[last_comma:]

                    remade_tokens = remade_tokens[:last_comma]
                    length = len(remade_tokens)

                    rem = int(math.ceil(length / 75)) * 75 - length
                    remade_tokens += [id_end] * rem + reloc_tokens
                    multipliers = multipliers[:last_comma] + [1.0] * rem + reloc_mults

                remade_tokens.append(token)
                multipliers.append(weight)
                i += 1

        token_count = len(remade_tokens)
        prompt_target_length = get_target_prompt_token_count(token_count)
        tokens_to_add = prompt_target_length - len(remade_tokens)

        remade_tokens = remade_tokens + [id_end] * tokens_to_add
        multipliers = multipliers + [1.0] * tokens_to_add

        return remade_tokens, multipliers, token_count

    def process_text(self, texts):
        used_custom_terms = []
        remade_batch_tokens = []
        token_count = 0
        
        cache = {}
        batch_multipliers = []
        for line in texts:
            if line in cache:
                remade_tokens, multipliers = cache[line]
            else:
                remade_tokens, multipliers, current_token_count = self.tokenize_line(line, used_custom_terms)
                token_count = max(current_token_count, token_count)
        
                cache[line] = (remade_tokens, multipliers)
            
            remade_batch_tokens.append(remade_tokens)
            batch_multipliers.append(multipliers)
        return batch_multipliers, remade_batch_tokens, used_custom_terms, token_count

    def process_tokens(self, remade_batch_tokens, batch_multipliers):
        self.device = self.text_encoder.device
        id_start = self.tokenizer.bos_token_id
        id_end = self.tokenizer.eos_token_id
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
        batch_multipliers, remade_batch_tokens, used_custom_terms, token_count = self.process_text(text)
        z = None
        i = 0
        while max(map(len, remade_batch_tokens)) != 0:
            rem_tokens = [x[75:] for x in remade_batch_tokens]
            rem_multipliers = [x[75:] for x in batch_multipliers]

            tokens = []
            multipliers = []
            for j in range(len(remade_batch_tokens)):
                if len(remade_batch_tokens[j]) > 0:
                    tokens.append(remade_batch_tokens[j][:75])
                    multipliers.append(batch_multipliers[j][:75])
                else:
                    tokens.append([self.tokenizer.eos_token_id] * 75)
                    multipliers.append([1.0] * 75)

            z1 = self.process_tokens(tokens, multipliers)
            z = z1 if z is None else torch.cat((z, z1), axis=-2)

            remade_batch_tokens = rem_tokens
            batch_multipliers = rem_multipliers
            i += 1

        return z