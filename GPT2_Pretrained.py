from dataclasses import dataclass
import torch 
import torch.nn as nn 
from torch.nn import functional as F 

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Model Architecture
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@dataclass
class GPTConfig: 
    block_size: int = 1024 
    vocab_size: int = 50257 
    n_layer: int = 12 
    n_head: int = 12 
    n_embd: int = 768 


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
 
        # n_embd must divide evenly across all heads, otherwise concatenating
        # the heads back together at the end won't reconstruct n_embd exactly
        assert config.n_embd % config.n_head == 0
 
         # key, query, value projections for all heads, but in a batch
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.n_embd // config.n_head  # size of each individual head's q/k/v vectors
 

        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
 
        # output projection: mixes information ACROSS heads after they've been computed independently and concatenated back together. Shape is already n_embd -> n_embd
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.proj.NANOGPT_SCALE_INIT = 1 # attaching an extra attribute

        # causal mask: a lower-triangular matrix of 1s and 0s, built ONCE at the maximum possible sequence length (block_size). 1 = "allowed to attend to this position",
        # 0 = "this is a future position, must be masked out". Stored as a buffer (not a parameter since it remains constant)
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
 
    def forward(self, x):
        # x.shape = (B, T, C) = (no.examples, block_size_used, n_embd)
        B, T, C = x.shape
 
        # apply linear projections to x to get the full-size key, query and value matrices
        k = self.key(x)    # k.shape = (B, T, n_embd)
        q = self.query(x)  # q.shape = (B, T, n_embd)
        v = self.value(x)  # v.shape = (B, T, n_embd)
 
        # split the last dimension (n_embd) into (n_head, head_size), then move n_head into the batch-like dimension via transpose so that matrix multiplication treats each head completely independently.
        # .view(...): (B, T, n_embd) -> (B, T, n_head, head_size)
        # .transpose(1, 2): (B, T, n_head, head_size) -> (B, n_head, T, head_size)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        # PyTorch has a trick that treats every dim except for the last 2 as batch dimensions 

        # Line-by-line Self-Attention
        # compute raw attention scores ("affinities") between every position's query and every position's key, independently within each head.
        "att = (q @ k.transpose(-2, -1)) * (self.head_size ** -0.5)"
        "att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))"
        "att = F.softmax(att, dim=-1)" 
        "y = att @ v"

        "Flash Attention Shortcut"
        # Flash Attention computes the exact same softmax(QK^T/sqrt(d))V result as the manual implementation above, 
        # but restructures the computation into tiles processed via a fused GPU kernel, using a running-softmax trick so the full (T,T) attention matrix is never fully materialized in slow VRAM.
        # This eliminates the memory read/write bottleneck of the standard approach (which is the real limiting factor, not raw compute), 
        # making it significantly faster and more memory-efficient without changing the mathematical output at all.
        # The math is identical, what changes is purely the computational strategy for executing that math on the GPU - specifically how data moves between fast on-chip memory and slower VRAM during the calculation
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
 
        # recombine all heads back into a single n_embd-sized vector per position.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
 
        # final output projection: mixes information across the (currently independent, just-concatenated) heads together into a genuinely blended representation.
        y = self.proj(y)
 
        return y
    

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # define the layers we want to implement int he MLP portion of the block 
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # gelu is just a relu that does not have a flat tail - gets rid of the dead neuron problem
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd) # this projection is what transforms the output back to the original shape (B, T, C) to allow for residual addition

        self.proj.NANOGPT_SCALE_INIT = 1 # attaching an extra attribute

    def forward(self, x):
        # implement the MLP as x flows through it
        # x passes through each individual layer, reassigns the value to x, and returns the final x 
        x = self.c_fc(x) 
        x = self.gelu(x) 
        x = self.proj(x) 
        return x


class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()

        # layernorm 1 
        # Keeps each token's value consistent and controlled (gaussian) at every layer, preventing the scale drift across depth that would cause softmax saturation and vanishing/exploding gradients
        self.ln_1 = nn.LayerNorm(config.n_embd) 
        # Self-attention
        # Enables tokens to communicate with one another
        self.attn = CausalSelfAttention(config) # different object separate for attention itself
        # layernorm 2
        self.ln_2 = nn.LayerNorm(config.n_embd) # second layernorm 
        # MLP/FNN
        # Takes what attention gathered (cross-token context) and processes it further, per token (no more cross-token mixing), via mathematical computations
        self.mlp = MLP(config) # different object separate for MLP
    
    # in a single forward pass of the loop perform the following layers in order: layernorm -> attention -> residual add -> layer norm -> feedforward -> residual add
    def forward(self, x):
        # residual addition (accumulated into x so the final output contains all oeprations from the transformer block)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x)) 
        return x
    

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            # word token embedding (embedding word tokens to vectors)
            wte = nn.Embedding(config.vocab_size, config.n_embd), # key=value is another way to setup a dict using dict() 
            # word position embedding (embedding position of word to vectors)
            wpe = nn.Embedding(config.block_size, config.n_embd),
            # stack of transformer blocks which consists of layernorm, multihead self-attention, residual add, MLP 
            # creates a list of instances of Block - 1 Block object for each loop of the block layer
            h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
            # layernorm layer
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self.__init_weights)

    # default initialisation
    # helps to guarantee that the weights and parameters upon initialisation produce reasonable losses that aren't inflated
    def __init_weights(self, module):
        if isinstance(module, nn.Linear): # if a Linear module
            std = 0.02 
            if hasattr(module, 'NANOGPT_SCALE_INIT'): # check if it has the manual tag - we only consider attn and mlp outputs since they are accumulated onto x, other outputs are not involved involved in residual additions to x
                std *= (2 * self.config.n_layer) ** -0.5 # if it has the tag then scale down the std by some variable amount (shrink the default 0.02)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # initialises the module's weights by randomly sampling values from a normal dist. with mean=0.0 and std scaled down
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding): # if an embedding module
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None): # a
        B, T = idx.shape
        # T cannot be greater than the blocK_size, that is the maximum length of the context window that we can provide
        assert T <= self.config.block_size, f"Cannot forward sequence length of {T}, block_size is only {self.config.block_size}"

        # forward the token and position embeddings
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(torch.arange(0, T, dtype=torch.long, device=idx.device)) # creates an embedding vector for each of the 0 to T positions in a sequence
        x = tok_emb + pos_emb # combines position and token embeddings to get a vector representetive of the meaning and position

        # forward the blocks for the transformer
        # transformer.h stores all the blocks and we are simply iterating over all the blocks and passing x through them
        for block in self.transformer.h:
            x = block(x)

        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x) 
        logits = self.lm_head(x) # softmax applied when calculating loss use F.cross_entropy
        loss = None 
        if targets is not None: 
            # flattens logits and targets from 3D to 2D and 1D respectively to calculate the loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    # this loads in the weights for GPT-2 from huggingface
    # this is not used for training and does not replace training, instead we laod these weights in to check that our model architecture works as intended and matches GPT-2
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface into our GPT class"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]

        # NOTE: our CausalSelfAttention uses 3 separate Linear layers (key, query, value, proj)
        # instead of HF's fused c_attn/c_proj, AND our MLP now uses "proj" instead of "c_proj".
        # None of these can be matched 1:1 by name against HF's keys, so we exclude all of them
        # here and handle every one manually further down.
        attn_keys = [k for k in sd_keys if '.attn.key.' in k or '.attn.query.' in k
                        or '.attn.value.' in k or '.attn.proj.' in k]
        mlp_proj_keys = [k for k in sd_keys if '.mlp.proj.' in k]
        sd_keys = [k for k in sd_keys if k not in attn_keys and k not in mlp_proj_keys]

        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]

        # same reasoning: c_attn/c_proj (attn) and mlp.c_proj on the HF side have no 1:1
        # counterpart in our naming, so exclude them here too and handle manually below
        attn_keys_hf = [k for k in sd_keys_hf if '.attn.c_attn.' in k or '.attn.c_proj.' in k]
        mlp_proj_keys_hf = [k for k in sd_keys_hf if '.mlp.c_proj.' in k]
        sd_keys_hf = [k for k in sd_keys_hf if k not in attn_keys_hf and k not in mlp_proj_keys_hf]

        transposed = ['mlp.c_fc.weight']  # mlp.c_proj removed -- now handled manually below
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        # --- manual attention weight loading ---
        # HF stores q, k, v fused as one Conv1D weight (n_embd, 3*n_embd) per block.
        # Our model keeps them as 3 separate Linear layers (n_embd, n_embd) each, so we split
        # HF's fused tensor into 3 equal chunks and route each into its matching layer,
        # transposing since Conv1D is (in, out) and Linear expects (out, in).
        for i in range(config.n_layer):
            w_hf = sd_hf[f'transformer.h.{i}.attn.c_attn.weight']
            b_hf = sd_hf[f'transformer.h.{i}.attn.c_attn.bias']

            w_q, w_k, w_v = w_hf.split(config.n_embd, dim=1)
            b_q, b_k, b_v = b_hf.split(config.n_embd, dim=0)

            with torch.no_grad():
                sd[f'transformer.h.{i}.attn.query.weight'].copy_(w_q.t())
                sd[f'transformer.h.{i}.attn.key.weight'].copy_(w_k.t())
                sd[f'transformer.h.{i}.attn.value.weight'].copy_(w_v.t())
                sd[f'transformer.h.{i}.attn.query.bias'].copy_(b_q)
                sd[f'transformer.h.{i}.attn.key.bias'].copy_(b_k)
                sd[f'transformer.h.{i}.attn.value.bias'].copy_(b_v)

            # attn.c_proj -> our attn.proj (1:1 shape, just needs the Conv1D transpose)
            w_proj_hf = sd_hf[f'transformer.h.{i}.attn.c_proj.weight']
            b_proj_hf = sd_hf[f'transformer.h.{i}.attn.c_proj.bias']
            with torch.no_grad():
                sd[f'transformer.h.{i}.attn.proj.weight'].copy_(w_proj_hf.t())
                sd[f'transformer.h.{i}.attn.proj.bias'].copy_(b_proj_hf)

            # mlp.c_proj -> our mlp.proj (same story, name-only mismatch)
            w_mlp_proj_hf = sd_hf[f'transformer.h.{i}.mlp.c_proj.weight']
            b_mlp_proj_hf = sd_hf[f'transformer.h.{i}.mlp.c_proj.bias']
            with torch.no_grad():
                sd[f'transformer.h.{i}.mlp.proj.weight'].copy_(w_mlp_proj_hf.t())
                sd[f'transformer.h.{i}.mlp.proj.bias'].copy_(b_mlp_proj_hf)

        return model
    

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=50):
    """
    Autoregressively generates tokens from the model.
    idx: (B, T) tensor of token indices — the current context
    Returns idx extended by max_new_tokens tokens: (B, T + max_new_tokens)
    """
    model.eval()

    for _ in range(max_new_tokens):
        # if the context grows beyond block_size, crop it to the last block_size tokens
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # forward the model to get logits for the next token
        logits, _ = model(idx_cond)  # logits.shape = (B, T, vocab_size)

        # only care about the logits at the last position (next-token prediction)
        logits = logits[:, -1, :] / temperature  # (B, vocab_size)

        # optionally crop to top-k tokens for sampling — keeps generation coherent
        # by zeroing out probability mass on unlikely tokens
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float('-inf')

        # convert logits to probabilities
        probs = F.softmax(logits, dim=-1)

        # sample the next token from the distribution (not greedy argmax,
        # so generation isn't deterministic / doesn't loop)
        idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

        # append sampled token to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    model.train()
    return idx