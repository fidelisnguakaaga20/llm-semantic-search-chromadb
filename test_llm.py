from transformers import pipeline

llm = pipeline("text-generation", model="gpt2")

result = llm(
    "Learning LLMs with confidence:",
    max_length=40,
    num_return_sequences=1
)

print(result[0]["generated_text"])
