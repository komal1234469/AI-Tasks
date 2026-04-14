# pip install tokenizers
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece
from tokenizers.trainers import BpeTrainer, WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
text = "Happiness is increasing"
print("Original Text:",text)
text=text.lower()
# 1. Whitespace
print("Whitespace:", text.split())

# 2. BPE
bpe = Tokenizer(BPE(unk_token="[UNK]"))
bpe.pre_tokenizer = Whitespace()
bpe.train_from_iterator([text], BpeTrainer(vocab_size=30))
print("BPE:", bpe.encode(text).tokens)

# 3. WordPiece
wp = Tokenizer(WordPiece(unk_token="[UNK]"))
wp.pre_tokenizer = Whitespace()
wp.train_from_iterator([text], WordPieceTrainer(vocab_size=30))
print("WordPiece:", wp.encode(text).tokens)