import torch
import torch.nn as nn
from typing import Any, Dict, List
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, use_swiglu = False):
        super().__init__()
        self.use_swiglu = use_swiglu
        if use_swiglu:
            self.linear1 = nn.Linear(d_model, 2 * d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
        else:
            self.linear1 = nn.Linear(d_model, d_ff)
            self.gelu = nn.GELU()
            self.linear2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.use_swiglu:
            x_proj = self.linear1(x)
            x1, x2 = torch.chunk(x_proj, 2, dim=-1)
            x = x1*F.silu(x2)
        else:
            x = self.gelu(self.linear1(x))
        return self.linear2(self.dropout(x))
    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, mode = "standard", tau=1.0, dropout = 0.1, use_swiglu=False):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads, mode=mode, tau=tau)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model, d_ff, dropout=dropout, use_swiglu=use_swiglu)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask):
        # Implementing feed-forward logic with batching.
        # Attention + residual
        attn_out = self.attention(self.norm1(x), attention_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward + residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, mode="standard", tau=1.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.mode = mode
        self.tau = tau

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def safe_softmax(self, tensor, dim):
        tensor = tensor - tensor.max(dim=dim, keepdim=True).values
        return torch.softmax(tensor, dim=dim)

    def forward(self, x, attention_mask=None):
        # x shape: (B, T, d_model)
        B, T, D = x.size()

        #Generate Q, K, V
        q_prime = self.q_proj(x)
        k_prime = self.k_proj(x)
        v_prime = self.v_proj(x)

        #Split into heads
        Q = q_prime.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = k_prime.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = v_prime.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        #Compute attention scores
        scale = 1.0 / (self.head_dim ** 0.5)
        scores = (Q @ K.transpose(-2, -1)) * scale

        if self.mode == "tanh-clipped":
            scores = self.tau*torch.tanh(scores)
        #Apply attention mask (if given)
        if attention_mask is not None:
            # attention_mask shape should be (B, T)
            # We expand it to match attention score dimensions
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # mask shape: (B, 1, 1, T)
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(T, T, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        #Softmax via safe softmax
        attention_weights = self.safe_softmax(scores, dim=-1)

        #Weighted sum
        output = attention_weights @ V
        #(B, num_heads, T, head_dim)

        #Combine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, D)

        #Final projection
        return self.out_proj(output)


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()
        self.config = config
        #print(config)
        #extract required config value
        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        num_heads = config["n_heads"]
        max_seq_len = 512

        d_ff = 4 * d_model
        mode = config.get("mode", "standard")
        tau = config.get("tau", 1.0)
        use_swiglu = config.get("use_swiglu", False)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, num_heads, d_ff, mode=mode, tau=tau, dropout=0.1, use_swiglu=use_swiglu) for _ in range(config["n_layers"])]
        )
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        #token embedding layer
        self.embedding = nn.Embedding(vocab_size,d_model)

        # output projection layer (maps hidden state -> vocab logits)
        self.output_projection = nn.Linear(d_model, vocab_size, bias = False)
        
        #Positional encoding (sinusoidal)
        pe = self._create_positional_encoding(max_seq_len, d_model)

        # Register as buffer (not trainable, but moves with model)
        self.register_buffer("positional_encoding", pe)

    
    def _create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model, dtype=torch.float32)

        position = torch.arange(0, max_seq_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        #print(weights.keys())
        with torch.no_grad():
            # Embedding
            self.embedding.weight.copy_(torch.as_tensor(weights["W_vocab"]).T)

            # output projection
            self.output_projection.weight.copy_(torch.as_tensor(weights["W_devocab"]).T)
            # transformer layer
            num_heads = self.config["n_heads"]
            for i, block in enumerate(self.blocks):
                layer = i + 1
            # multi-head Q K V
                q_weights = []
                k_weights = []
                v_weights = []

                for h in range(num_heads):
                    head = h + 1
                    q_weights.append(torch.as_tensor(weights[f"W_{layer}_Q_{head}"]))
                    k_weights.append(torch.as_tensor(weights[f"W_{layer}_K_{head}"]))
                    v_weights.append(torch.as_tensor(weights[f"W_{layer}_V_{head}"]))
                
                q_matrix = torch.cat(q_weights, dim=0)
                k_matrix = torch.cat(k_weights, dim=0)
                v_matrix = torch.cat(v_weights, dim=0)

                block.attention.q_proj.weight.copy_(q_matrix.T)
                block.attention.k_proj.weight.copy_(k_matrix.T)
                block.attention.v_proj.weight.copy_(v_matrix.T)

                # Attention output projection
                block.attention.out_proj.weight.copy_(
                    torch.as_tensor(weights[f"W_{layer}_O"]).T
                )

                # Feedforward
                block.ffn.linear1.weight.copy_(
                    torch.as_tensor(weights[f"W_{layer}_up"]).T
                )

                block.ffn.linear1.bias.copy_(
                    torch.as_tensor(weights[f"b_{layer}_up"])
                )

                block.ffn.linear2.weight.copy_(
                    torch.as_tensor(weights[f"W_{layer}_down"]).T
                )

                block.ffn.linear2.bias.copy_(
                    torch.as_tensor(weights[f"b_{layer}_down"])
                )

                # LayerNorm 1
                block.norm1.weight.copy_(
                    torch.as_tensor(weights[f"gamma_{layer}_1"])
                )
                block.norm1.bias.copy_(
                    torch.as_tensor(weights[f"beta_{layer}_1"])
                )

                # LayerNorm 2
                block.norm2.weight.copy_(
                    torch.as_tensor(weights[f"gamma_{layer}_2"])
                )
                block.norm2.bias.copy_(
                    torch.as_tensor(weights[f"beta_{layer}_2"])
                )
            
            if "gamma_final" in weights:
                self.final_norm.weight.copy_(torch.as_tensor(weights["gamma_final"]))
                self.final_norm.bias.copy_(torch.as_tensor(weights["beta_final"]))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """
        #convert token-ids to embeddings
        x = self.embedding(input_ids)

        seq_len = input_ids.size(1)

        # Add positional encoding
        pos = self.positional_encoding[:seq_len, :]
        x = x + pos.unsqueeze(0)

        for block in self.blocks:
            x = block(x, attention_mask)

        #project hidden state to vocabulary state
        x = self.final_norm(x)
        logits = self.output_projection(x)
        return logits



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    
    input_ids_list = batch["input_ids"]
    attention_mask_list = batch["attention_mask"]

    batch_size = len(input_ids_list)
    
    max_len = max(seq.size(0) for seq in input_ids_list)

    padded_input_ids = torch.full(
        (batch_size, max_len),
        fill_value = PAD_ID,
        dtype = input_ids_list[0].dtype
    )

    padded_attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype = attention_mask_list[0].dtype
    )

    for i in range(batch_size):
        seq_len = input_ids_list[i].size(0)

        padded_input_ids[i, :seq_len] = input_ids_list[i]
        padded_attention_mask[i, :seq_len] = attention_mask_list[i]

    return {
        "input_ids" : padded_input_ids,
        "attention_mask" : padded_attention_mask
    }
