from collections import defaultdict
import re

class BPETokenizer:
    def __init__(self, vocab_size=None, special_tokens=None):
        self.vocab_size = vocab_size

        if special_tokens is None:
            special_tokens = ["[PAD]", "[UNK]", "[SOS]", "[EOS]"]
        
        self.special_tokens = special_tokens

        #vocabulary containers
        self.vocab = set()
        self.token_to_id = {}
        self.id_to_token = {}

        #BPE merge rules
        self.merges = []

        #UNKNOWN token
        self.unk_token = "[UNK]"
    
    
    def _atomic_merge_step(self, word_list, pair, new_token):
        merged = []
        i = 0
        a, b = pair
        n = len(word_list)
        while i < n:
            if i < n-1 and word_list[i]==a and word_list[i+1]==b:
                merged.append(new_token)
                i += 2
            else:
                merged.append(word_list[i])
                i += 1
        return tuple(merged)


    def train(self, corpus):
        word_freq = defaultdict(int)
        #Build word frequency dictionary using characters tokens
        for sentence in corpus:
            words = sentence.split()
            for i, word in enumerate(words):
                token = (" " + word) if i > 0 else word
                chars = tuple(token)
                word_freq[chars] += 1
        
        # character vocabulary
        for word in word_freq:
            for char in word:
                self.vocab.add(char)
        self.vocab.add("\u200c")
        self.vocab.add("\u200d") 

        # add special tokens to vocab
        for token in self.special_tokens:
            self.vocab.add(token)
        
        # assign ids to special tokens first
        for token in self.special_tokens:
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        #initializing the token ids for base vocabulary
        for token in sorted(self.vocab):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
        
        num_merges = self.vocab_size - len(self.token_to_id)
        # Begin iterative vocabulary construction.
        for _ in range(max(0,num_merges)):
            digram_stats = defaultdict(int)

            #count pair frequency
            for tokens, freq in word_freq.items():
                for a, b in zip(tokens, tokens[1:]):
                    digram_stats[(a,b)]+=freq
            
            #stop if no more pair exists
            if not digram_stats:
                break

            #select most frequent pair
            best_pair = max(digram_stats.items(), key=lambda x: (x[1], x[0]))[0]
            new_token = best_pair[0] + best_pair[1]

            # add new token to vocabulary
            self.vocab.add(new_token)

            #store merge rule
            self.merges.append((best_pair, new_token))
            #Apply merge to all words in corpus
            new_word_freq = defaultdict(int)
            for tokens, freq in word_freq.items():
                merged_tokens = self._atomic_merge_step(tokens, best_pair, new_token)
                new_word_freq[merged_tokens]+=freq
            word_freq = new_word_freq
        
        #update token ids for newly created tokens
        for token in sorted(self.vocab):
            if token not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[token] = idx
                self.id_to_token[idx] = token
            
    
    
    def encode(self, text):
        # Begin encoding the user given input text.
        output_tokens = []
        words = text.split()
        tokens = []
        for i, word in enumerate(words):
            token = (" "+word) if i > 0 else word
            tokens.extend(list(token))
        
            
        #apply merges in learned order
        for pair, new_token in self.merges:
            tokens = self._atomic_merge_step(tokens, pair, new_token)
            
        #convert tokens to ids
        for token in tokens:
            if token in self.token_to_id:
                output_tokens.append(self.token_to_id[token])
            else:
                output_tokens.append(self.get_unk_id())
        return output_tokens


    def decode(self, token_ids):
        # Begin decoding the user given input text.
        tokens = []
        for idx in token_ids:
            token = self.id_to_token.get(idx, self.unk_token)
            if token not in self.special_tokens: #ignore the padding
                tokens.append(token)
        return "".join(tokens)

    def save(self, filepath):
        import json 
        import os

        os.makedirs(filepath, exist_ok=True)
        data = {
            "vocab": list(self.vocab),
            "token_to_id": self.token_to_id,
            "id_to_token": self.id_to_token,
            "merges": self.merges,
            "special_tokens": self.special_tokens
        }

        with open(os.path.join(filepath, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        import json
        import os

        with open(os.path.join(filepath, "tokenizer.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

        self.vocab = set(data["vocab"])
        self.token_to_id = data["token_to_id"]

        # JSON converts keys to strings, so convert them back to int
        self.id_to_token = {int(k): v for k, v in data["id_to_token"].items()}
        self.merges = [tuple(pair) for pair in data["merges"]]
        self.special_tokens = data["special_tokens"]
        
    
    def get_vocab_size(self):
        return len(self.token_to_id)
    
    def get_unk_id(self):
        return self.token_to_id[self.unk_token]
