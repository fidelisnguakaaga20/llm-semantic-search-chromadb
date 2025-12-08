# 04_transformer_concepts.py
# Week 4 – Transformers Concepts (Tokenization, Attention, Decoder-style model)

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "gpt2"

print("Loading tokenizer and model:", MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    output_attentions=True,   # <<< we want attention weights
)

# ---------------------------------------------------------
# 1. TOKENIZATION (DEEPER LOOK)
# ---------------------------------------------------------
print("\n=== 1) TOKENIZATION ===\n")

text_1 = "Learning LLMs with confidence."
text_2 = "Learning large language models with confidence."

enc_1 = tokenizer(text_1, return_tensors="pt")
enc_2 = tokenizer(text_2, return_tensors="pt")

print("Text 1:", text_1)
print("Tokens 1:", tokenizer.convert_ids_to_tokens(enc_1["input_ids"][0].tolist()))
print("IDs 1   :", enc_1["input_ids"][0].tolist())
print()

print("Text 2:", text_2)
print("Tokens 2:", tokenizer.convert_ids_to_tokens(enc_2["input_ids"][0].tolist()))
print("IDs 2   :", enc_2["input_ids"][0].tolist())
print()

print("Note how uncommon / longer words may be split into multiple sub-tokens.\n")

# ---------------------------------------------------------
# 2. ATTENTION (BASIC)
# ---------------------------------------------------------
print("\n=== 2) ATTENTION (BASIC) ===\n")

sample_text = "Nguakaaga is learning LLM engineering with Python."
inputs = tokenizer(sample_text, return_tensors="pt")

print("Sample text:", sample_text)
print("Tokens:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist()))
print("Sequence length:", inputs["input_ids"].shape[1])

with torch.no_grad():
    outputs = model(**inputs)

# outputs.attentions is a tuple: (num_layers, batch, num_heads, seq_len, seq_len)
attentions = outputs.attentions
num_layers = len(attentions)
batch_size, num_heads, seq_len, _ = attentions[-1].shape

print(f"\nNumber of layers: {num_layers}")
print(f"Last layer attention shape: (batch={batch_size}, heads={num_heads}, seq_len={seq_len}, seq_len={seq_len})")

# Let's inspect attention for:
# - last layer
# - head 0
# - last token attending to all previous tokens
last_layer_attn = attentions[-1]            # shape: (batch, heads, seq_len, seq_len)
head_0 = last_layer_attn[0, 0]              # shape: (seq_len, seq_len)
last_token_attn = head_0[-1]                # shape: (seq_len,)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0].tolist())

print("\nAttention from LAST token to all tokens (last layer, head 0):")
for tok, weight in zip(tokens, last_token_attn.tolist()):
    print(f"  token={tok:15}  weight={weight:.4f}")

print("\nHigher weights ≈ the tokens this head is focusing on when predicting the last token.\n")

# ---------------------------------------------------------
# 3. DECODER (GPT-STYLE) – HIGH LEVEL SUMMARY
# ---------------------------------------------------------
print("\n=== 3) DECODER MODEL (GPT-STYLE) – SUMMARY ===\n")

decoder_summary = """
GPT-style transformer (like gpt2) is a stack of decoder blocks:

1) Input Embedding:
   - Each token ID is mapped to a vector (embedding).
   - Positional embeddings are added so the model knows order.

2) Masked Self-Attention:
   - Each token looks at previous tokens (and itself), not future ones.
   - 'Masked' = can't peek ahead, so it generates autoregressively.

3) Feed-Forward Network (MLP):
   - After attention, each position goes through a small neural network
     (same network shared across all positions).

4) Residual + LayerNorm:
   - Each block has residual connections and layer normalization
     to help training stability.

5) Stack many layers:
   - gpt2 has multiple such decoder blocks stacked.
   - Final layer outputs logits for the next-token prediction.

In our code above, when we call model(**inputs), all of this runs
under the hood. We just see:
   - logits (for next tokens)
   - attentions (for each layer and head).
"""

print(decoder_summary)
print("\n[Week 4] Tokenization, attention, and decoder concepts demonstrated.\n")
