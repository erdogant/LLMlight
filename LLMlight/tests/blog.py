# =============================================================================
# # %% code block 1
# =============================================================================

from LLMlight import LLMlight

client = LLMlight(
          chunks={'type': 'words', 'size': 250, 'n': 5},
          embedding='bert',
          retrieval_method='naive_rag',
          context_strategy='chunk-wise',
          model="gpt-oss-20b",
          endpoint="http://localhost:1234/v1/chat/completions",
)

# Check available models
print(client.get_available_models(validate=False))

# Article
article_text = client.read_pdf('https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf')

# Test the connection with a simple question:
response = client.prompt(
    query="What is the main goal that is achieved in this article?",
    context=article_text,
    instructions="Answer in two concise sentences.",
    temperature=0
)

print(response)

# %%
# =============================================================================
# 
# =============================================================================

