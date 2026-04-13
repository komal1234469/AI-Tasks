from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = WordPieceTrainer(vocab_size=38)

data = [
    "i love playing cricket",
    "he is playing football",
    "they played well"
]

tokenizer.train_from_iterator(data, trainer)

output = tokenizer.encode("playing cricket")
print(output.tokens)

