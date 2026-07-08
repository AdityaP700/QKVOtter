import regex as re
import pickle
from collections import defaultdict
import tiktoken
import sentencepiece as spm
import json
import os

class BasicTokenizer:
    def __init__(self):
        self.merges = {}   # (int, int) -> int
        self.vocab = {}    # int -> bytes
#scans a list of token iDs and counts how often every adjacent
#pair of token appears together
    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts

#takes a specific pair of tokens,which might be 'e' and 'n'
#replaces every occurence of that pair in the text with
#a brand new token ID
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

#starts with a base vocab of 256 individual bytes ,
#it then loops vocab-256 times ,finding the
#most frequent pairs of tokens
#adding the new merged tokens to the vocabulary
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        tokens = list(text.encode("utf-8"))
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            tokens = self.merge(tokens, pair, idx)
            self.merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 1000 == 0:
                print(f"Merge {i}/{num_merges}: {pair} -> {idx}")

#to encode it ,takes text ,turns it into bytes ,and keeps applying
#learned merges until no more merges can be found

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            # Get the pair with the lowest merge rank (earliest learned)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

#to decode ,it looks up the byte sequences for each token id and
#joins for each token ID and joins them back into text
    def decode(self, ids):
        tokens = b"".join(self.vocab.get(idx, b"\ufffd") for idx in ids)  # � for unknown
        return tokens.decode("utf-8", errors="replace")

    def save(self, file_path):
        data = {
            "merges": self.merges,
            "vocab": {k: v for k, v in self.vocab.items()}  # keep bytes
        }
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved to {file_path}")

    def load(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.merges = data["merges"]
        self.vocab = data["vocab"]
        print(f"Loaded from {file_path} | Vocab: {len(self.vocab)} | Merges: {len(self.merges)}")

# the flaw with the basic tokenizer is that ,it may not able to distinguish when to split or merge effectively

class RegexTokenizer(BasicTokenizer):
    def __init__(self):
        super().__init__()
        #its similar as the pattern being used in openAI
        self.pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
        self.compiled = re.compile(self.pattern)

#instead of passing the whole document to the BPE engine
#it first splits the document into a list of chunks
    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        chunks = self.compiled.findall(text)
        chunk_tokens = [list(ch.encode("utf-8")) for ch in chunks]

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        num_merges = vocab_size - 256
        merges = {}
# it then runs the get_stats() inside each chunk independently
#it ensure that bpe merger can never cross a chunk boundary
        for i in range(num_merges):
            stats = defaultdict(int)
            for ct in chunk_tokens:
                for pair, cnt in self.get_stats(ct).items():
                    stats[pair] += cnt
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            chunk_tokens = [self.merge(ct, pair, idx) for ct in chunk_tokens]
            merges[pair] = idx
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and i % 1000 == 0:
                print(f"Merge {i}/{num_merges}")

        self.merges = merges

    def encode(self, text):
        chunks = self.compiled.findall(text)
        ids = []
        for ch in chunks:
            tokens = list(ch.encode("utf-8"))
            while len(tokens) >= 2:
                stats = self.get_stats(tokens)
                pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                tokens = self.merge(tokens, pair, idx)
            ids.extend(tokens)
        return ids

#tiktoken implementation
class TiktokenTokenizer:
    def __init__(self, model_name="cl100k_base"):
        self.enc = tiktoken.get_encoding(model_name)

    def process(self, text):
        ids = self.enc.encode(text)
        pieces = [self.enc.decode([idx]) for idx in ids]
        return {
            'ids': ids,
            'pieces': pieces,
            'count': len(ids)
        }

class SentencePieceTokenizer:
    #it says the sentencepiece to use Byte Pair encoding style
    def __init__(self, vocab_size=400, model_type="bpe"):
        #it needs a minimum vocab size to account for all base
        #characters ,byte fallbacks
        #special tokens
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.spm_input_file = "spm_input.txt"
        self.spm_model_prefix = "spm_model"
        self.sp = None

#
    def train(self, text):
        with open(self.spm_input_file, "w", encoding="utf-8") as f:
            f.write(text + "\n" + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()")
        try:
            #this acts as a factory
            spm.SentencePieceTrainer.train(
                input=self.spm_input_file,
                model_prefix=self.spm_model_prefix,
                vocab_size=self.vocab_size,
                model_type=self.model_type,
                byte_fallback=True
            )
            #this creates an empty "worker" object . right now it doesnt know how to define
            self.sp = spm.SentencePieceProcessor()
            #this line takes the .model file the factory just built and downloads it into the workers
            
            self.sp.load(f'{self.spm_model_prefix}.model')
        except Exception as e:
            print(f"SentencePiece training failed: {e}")
            self.sp = None
        finally:
            self.cleanup()

    def cleanup(self):
        for file in [self.spm_input_file, f"{self.spm_model_prefix}.model", f"{self.spm_model_prefix}.vocab"]:
            if os.path.exists(file):
                os.remove(file)

    def process(self, text):
        if not self.sp:
            return {'ids': [], 'pieces': [], 'count': 0}
        ids = self.sp.encode_as_ids(text)
        pieces = self.sp.encode_as_pieces(text)
        return {
            'ids': ids,
            'pieces': pieces,
            'count': len(ids)
        }

input_text = "I'm building chunkdUp in RUST"

tok = RegexTokenizer()

print("--- My BPE ---")
vocab_size = 256 + 10 # 10 merges
tok.train(input_text, vocab_size, verbose=False)

my_bpe_ids = tok.encode(input_text)
my_bpe_pieces = [tok.decode([idx]) for idx in my_bpe_ids]
my_bpe_count = len(my_bpe_ids)

print(f"Token IDs: {my_bpe_ids}")
print(f"Token Pieces: {my_bpe_pieces}")
print(f"Token count: {my_bpe_count}")

# Process with Tiktoken
print("\n--- Tiktoken (cl100k_base) ---")
tik_tok = TiktokenTokenizer()
tik_res = tik_tok.process(input_text)
print(f"Token IDs: {tik_res['ids']}")
print(f"Token Pieces: {tik_res['pieces']}")
print(f"Token count: {tik_res['count']}")

# Process with SentencePiece
print("\n--- SentencePiece ---")
spm_tok = SentencePieceTokenizer()
spm_tok.train(input_text)
spm_res = spm_tok.process(input_text)
print(f"Token IDs: {spm_res['ids']}")
print(f"Token Pieces: {[p.encode('ascii', 'replace').decode('ascii') for p in spm_res['pieces']]}")
print(f"Token count: {spm_res['count']}")

final_results = {
    "input_text": input_text,
    "my_bpe": {
        "ids": my_bpe_ids,
        "pieces": my_bpe_pieces,
        "count": my_bpe_count
    },
    "tiktoken": tik_res,
    "sentencepiece": spm_res
}

# Write results to the requested path
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "lab-001-tokenizer"))
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "experiment-001.json")

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=4)
print(f"\nResults stored in {output_file}")
