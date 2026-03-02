import torch
import torch.nn as nn
from typing import Any, Dict, List


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()
        
        #extract required config value
        vocab_size = config["vocab_size"]
        d_model = config["d_model"]
        max_seq_len = config["max_seq_len"]
        
        #token embedding layer
        self.embedding = nn.Embedding(vocab_size,d_model)

        # output projection layer (maps hidden state -> vocab logits)
        self.output_projection = nn.Linear(d_model, vocab_size, bias = False)
        
        #Positional encoding (sinusoidal)
        pe = self._create_positional_encoding(max_seq_len, d_model)

        # Register as buffer (not trainable, but moves with model)
        self.register_buffer("positional_encoding", pe)
    
    def _create_positional_encoding(self, max_seq_len, d_model):
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len).unsqueeze(1)

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
        raise NotImplementedError("Implement set_weights as described in assignment document")

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
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)

        #project hidden state to vocabulary state
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
