class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.vocab_size = vocab_size

        if special_tokens is None:
            special_tokens = ["[PAD]","[UNK]"]
        
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
        while i < len(word_list):
            if i < len(word_list)-1 and word_list[i]==pair[0] and word_list[i+1]==pair[1]:
                merged.append(new_token)
                i = i + 2
            else:
                merged.append(word_list[i])
                i = i + 1
        return merged


    def train(self, corpus):
        word_freq = {}
        #Build word frequency dictionary using characters tokens
        for sentence in corpus:
            words = sentence.strip().split()
            for word in words:
                chars = tuple(list(word))
                if chars not in word_freq:
                    word_freq[chars] = 0
                word_freq[chars]+=1
        
        # character vocabulary
        for word in word_freq:
            for char in word:
                self.vocab.add(char)

        # add special tokens to vocab
        for token in self.special_tokens:
            self.vocab.add(token)
        
        #initializing the token ids for base vocabulary
        for token in sorted(self.vocab):
            idx = len(self.token_to_id)
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token
        
        # iterative vocabulary construction
        while(len(self.vocab)<self.vocab_size):
            digram_stats = {}

            #count pair frequency
            for word, freq in word_freq.items():
                tokens = list(word)
                for i in range(len(tokens)-1):
                    pair = (tokens[i],tokens[i+1])
                    digram_stats[pair] = digram_stats.get(pair,0) + freq
            
            #stop if no more pair exists
            if not digram_stats:
                break

            #select most frequent pair
            best_pair = max(digram_stats, key=digram_stats.get)
            new_token = best_pair[0] + best_pair[1]

            # add new token to vocabulary
            self.vocab.add(new_token)

            #store merge rule
            self.merges.append(best_pair)
            #Apply merge to all words in corpus
            new_word_freq = {}
            for word, freq in word_freq.items():
                tokens = list(word)
                merged_tokens = self._atomic_merge_step(tokens, best_pair, new_token)
                merged_tokens = tuple(merged_tokens)
                if merged_tokens not in new_word_freq:
                    new_word_freq[merged_tokens] = 0
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
        words = text.strip().split()
        
        for word in words:
            tokens = list(word)
            
            #apply merges in learned order
            for pair in self.merges:
                new_token = pair[0] + pair[1]
                #check if pair exist in current tokens
                i = 0
                pair_found = False
                while i < len(tokens)-1:
                    if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                        pair_found = True
                        break
                    else:
                        i = i + 1
                if pair_found:
                    tokens = self._atomic_merge_step(tokens, pair, new_token)
            
            #convert tokens to ids
            for token in tokens:
                if token in self.token_to_id:
                    output_tokens.append(self.token_to_id[token])
                else:
                    output_tokens.append(self.get_unk_id())
        return output_tokens


    def decode(self, token_ids):
        print("Decode not implemented yet")

    def save(self, filepath):
        print("save not implemented yet")

    def load(self, filepath):
        print("Load method not implemented yet.")
    
    def get_vocab_size(self):
        return len(self.token_to_id)
    
    def get_unk_id(self):
        return self.token_to_id.get(self.unk_token)


#1.test 

# tokenizer = BPETokenizer(vocab_size=30)

# corpus = [
#     "lowest",
#     "lower",
#     "newest",
#     "widest"
# ]

# tokenizer.train(corpus)

# print("VOCAB:", tokenizer.vocab)
# print("MERGES:", tokenizer.merges)
# print(tokenizer.encode("lowest"))
# print(tokenizer.encode("lower"))
# print(tokenizer.encode("newest"))
# print(tokenizer.encode("widest"))
# print(tokenizer.encode("rohit"))
