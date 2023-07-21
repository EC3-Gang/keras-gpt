"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass

import numpy as np

import keras_core as keras
from keras_core import Model
from keras_core import layers
from keras_core.layers import Layer
from keras_core import ops
from keras_core import activations
from keras_core import initializers
from keras_core import losses
from keras_core import optimizers

def get_param(model):
    return model.count_params()

def np_multinomial(probs, num_samples=1):
    return np.array([np.random.choice(len(probs), p=probs) for _ in range(num_samples)])

class Inits:
    def __init__(self, config):
        self.weight = initializers.RandomNormal(mean=0.0, stddev=0.02)
        self.bias = initializers.Zeros()
        self.scaled = initializers.RandomNormal(mean=0.0, stddev=0.02/math.sqrt(2 * config.n_layer ))

class LayerNorm(Layer):
    def __init__(self, ndim, use_bias):
        super().__init__()
        self.normalize = layers.LayerNormalization(epsilon=1e-6)
        self.weight = self.add_weight(shape=ndim,
                                      initializer='ones',
                                      trainable=True)
        self.bias = self.add_weight(shape=ndim,
                                    initializer='zeroes',
                                    trainable=True) if use_bias else None

    def call(self, inputs):
        output = self.normalize(inputs)
        if self.bias is not None:
            output = output * self.weight + self.bias
        return output

class CausalSelfAttention(Layer):
    def __init__(self, config, weight_initializer, bias_initializer, scaled_initializer):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = layers.Dense(3 * config.n_embd, use_bias=config.bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
        self.c_proj = layers.Dense(config.n_embd, use_bias=config.bias, kernel_initializer=scaled_initializer, bias_initializer=bias_initializer)
        self.attn_dropout = layers.Dropout(config.dropout)
        self.resid_dropout = layers.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.bias = ops.triu(ops.ones((config.block_size, config.block_size)), -1).reshape(1, 1, config.block_size, config.block_size)


    def call(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = ops.split(qkv, indices_or_sections=3, axis=2)
        k = ops.transpose(ops.reshape(k, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)
        q = ops.transpose(ops.reshape(q, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)
        v = ops.transpose(ops.reshape(v, (B, T, self.n_head, C // self.n_head)), (0, 2, 1, 3)) # (B, nh, T, hs)
        att = ops.matmul(q, ops.transpose(k, axes=(0, 1, -1, -2))) * (1.0 / ops.sqrt(ops.size(k[-1])))
        att = ops.where(ops.expand_dims(ops.expand_dims(self.bias[:, :, :T, :T], axis=0), axis=1) == 0, float("-inf"), att)
        att = ops.softmax(att, axis=-1)
        att = self.attn_dropout(att)
        y = ops.matmul(att, v)  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = ops.transpose(y, axes=(0, 2, 1, 3, 4, 5))
        y = ops.reshape(y, (B, T, C))

        y = self.c_proj(y)
        return y


class MLP(Layer):

    def __init__(self, config, weight_initializer, bias_initializer, scaled_initializer):
        super().__init__()
        self.c_fc    = layers.Dense(4 * config.n_embd, use_bias=config.bias, kernel_initializer=weight_initializer, bias_initializer=bias_initializer)
        self.gelu    = ops.gelu
        self.c_proj  = layers.Dense(config.n_embd, use_bias=config.bias, kernel_initializer=scaled_initializer, bias_initializer=bias_initializer)
        self.dropout = layers.Dropout(config.dropout)

    def call(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(Layer):

    def __init__(self, config, weight_initializer, bias_initializer, scaled_initializer):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.attn = CausalSelfAttention(config, weight_initializer=weight_initializer, bias_initializer=bias_initializer, scaled_initializer=scaled_initializer)
        self.ln_2 = LayerNorm(config.n_embd, use_bias=config.bias)
        self.mlp = MLP(config, weight_initializer=weight_initializer, bias_initializer=bias_initializer, scaled_initializer=scaled_initializer)

    def call(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class Transformer(Layer):
    def __init__(self, config, weight_initializer, bias_initializer, scaled_initializer):
        super(Transformer, self).__init__()
        self.wte = layers.Embedding(config.vocab_size, config.n_embd, embeddings_initializer=weight_initializer)
        self.wpe = layers.Embedding(config.block_size, config.n_embd)
        self.drop = layers.Dropout(config.dropout)
        self.h = [Block(config, weight_initializer=weight_initializer, bias_initializer=bias_initializer, scaled_initializer=scaled_initializer) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd, use_bias=config.bias)

class GPT(Model):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.init = Inits(config)
        self.config = config
        self.transformer = Transformer(config, weight_initializer=self.init.weight, bias_initializer=self.init.bias, scaled_initializer=self.init.scaled)
        self.lm_head = layers.Dense(config.vocab_size, use_bias=False, kernel_initializer=self.init.weight, bias_initializer=self.init.bias)
        for emb in self.get_layers(layers.Embedding):
            emb.set_weights(self.lm_head.get_weights()) # https://paperswithcode.com/method/weight-tying

    def get_layers(self, layer_type):
        layers = [layer for layer in self.layers if isinstance(layer, layer_type)]
        return layers
    def call(self, idx, targets=None):
        b, t = idx.shape[0], idx.shape[1]
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = ops.arange(0, t) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = losses.CategoricalCrossentropy(from_logits=True)(targets, logits)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.get_layer('wpe').set_weights([ self.transformer.get_layer('wpe').get_weights()[0][:block_size] ])
        for block in self.transformer.h:
            if 'bias' in block.get_weights():
                bias_weight = block.get_weights()['bias']
                bias_weight = bias_weight[:,:,:block_size,:block_size]
                block.set_weights([bias_weight])

    @classmethod
    def from_pretrained(model_type, override_args=None):
        from transformers import TFGPT2LMHeadModel
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args)
        
        print("Loading weights from pretrained gpt: %s" % model_type)

        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024  # always 1024 for GPT model checkpoints
        config_args['bias'] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if 'dropout' in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']
        
        model_hf = TFGPT2LMHeadModel.from_pretrained(model_type)
        model_hf_weights = model_hf.weights

        # get model weights
        model_weights = model_hf_weights 
        model_weights = [w for w in model_weights if 'attn.bias' not in w.name]  # discard this mask / buffer, not a param

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(model_weights) == len(model_hf_weights), f"mismatched keys: {len(model_weights)} != {len(model_hf_weights)}"
        
        for w in model_weights:
            if any(w.name.endswith(t) for t in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert ops.transpose(model_hf.get_layer(w.name).get_weights()[0]).shape == w.shape
                w.assign(ops.transpose(model_hf.get_layer(w.name).get_weights()[0]))
            else:
                assert model_hf.get_layer(w.name).get_weights()[0].shape == w.shape
                w.assign(model_hf.get_layer(w.name).get_weights()[0])

        return model_hf

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = optimizers.AdamW(learning_rate=learning_rate, beta_1=betas[0], beta_2=betas[1], weight_decay=weight_decay)
        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = ops.top_k(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = activations.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = np_multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = ops.concatenate((idx, idx_next), dim=1)

        return idx