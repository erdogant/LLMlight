from LLMlight import LLMlight

client = LLMlight(endpoint="http://localhost:1234/v1/chat/completions")

# Check available models
print(client.get_available_models(validate=False))

# Article
article_text = client.read_pdf('https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf')

# Test the connection with a question:
# response = client.prompt(
#    query="What is the main goal that is achieved in this article?",
#    context=article_text,
#    instructions="Answer in two concise sentences.",
#    temperature=0
# )
# print(response)

# %%
# =============================================================================
# # %% code block 1
# =============================================================================
client_rubric = LLMlight(
          chunks={'type': 'words', 'size': 250, 'n': 5},
          embedding='bert',
          retrieval_method='naive_rag',
          context_strategy='chunk-wise',
          model="gpt-oss-20b",
          endpoint="http://localhost:1234/v1/chat/completions",
)

RUBRIC_QUESTIONS = [
    "Does it describe a feasible research methodology?",
    "Does it include a scientifically rigorous evaluation strategy?",
    "Does it include a model comparison?",
    "Does it describe an analysis of errors?"
]

def agent_rubric(client, question, document, TEMPERATURE):
    return client.prompt(
        query=question,
        context=document,
        system=(
            "You are a critical professor in data science. "
            "Evaluate the article using only its content."
        ),
        instructions=(
            "Answer the question directly. Do not invent facts or use external knowledge. "
            "Provide a supporting quote from the document when possible."
        ),
        temperature=TEMPERATURE
    )



# %%
# =============================================================================
# Agents
# =============================================================================
client_score = LLMlight(
          embedding='bert',
          retrieval_method=None,
          context_strategy=None,
          model="gpt-oss-20b",
          endpoint="http://localhost:1234/v1/chat/completions",
)


def agent_score(client, question, answer, TEMPERATURE=0):
    return client.prompt(
        query=f"Context:Question: {question}",
        context=f"\n\nAnswer: {answer}",
        system="You need to score using only the input of the feedback.",
        instructions=(
            "The rubric agent analyzed the article and explains his judgment in the context."
            "It is your task to return the score and a short justification.\n"
            "Score from 1 to 10 based on only the given context.\n"
            "Example: If the context describes 'No - the proposal does not contain ...' it means a relative lower score than 'Yes, the proposal contains ...' "
            "- **1-3:** No - the proposal does not contain..\n"
            "- **4-5:** No - the proposal does not contain.. but with improvements it can be more clear\n"
            "- **6-7:** Yes - but..\n"
            "- **8-9:** Yes - well-explained, clear, and complete but could be slightly refined.\n"
            "- **10:** Yes, the proposal is perfectly clear\n"
        ),

        response_format = (
            """ JSON object:
            {'score':<INTEGER>, 'justification':'<STRING>'}
            """
            ),
        temperature=TEMPERATURE,
    )

# %%
import json

results = []
for question in RUBRIC_QUESTIONS:
    # Get the answer from the rubric agent
    answer = agent_rubric(client_rubric, question, article_text, 0.8)

    # Get the score
    out = agent_score(client_score, question, answer)
    out = json.loads(out)

    # Store results
    results.append({
        "question": question,
        "answer": answer,
        "score": out['score'],
        "justification": out['justification'],
    })
    print(f"\nQ: {question}")
    print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
    print(f"Score: {out['score']}")

# %%

client_advisor = LLMlight(
          embedding='bert',
          retrieval_method=None,
          context_strategy=None,
          model="gpt-oss-20b",
          endpoint="http://localhost:1234/v1/chat/completions",
)

def agent_advisor(client, results):
    feedback = "\n\n".join(
        f"Question: {item['question']}\n"
        f"Assessment: {item['answer']}\n"
        f"Score: {item['score']}\n"
        f"Critical review: {item.get('critique', 'N/A')}"
        for item in results
    )

    return client.prompt(
        query="Prepare a structured feedback report.",
        context=feedback,
        system=(
            "You are an academic advisor preparing constructive feedback."
        ),
        instructions=(
            "Organize the feedback by rubric question. "
            "Highlight missing elements, explain what needs "
            "improvement, and distinguish evidence from "
            "recommendations. Do not introduce new factual "
            "claims or treat model-generated scores as verdicts."
        ),
        temperature=0
    )

report = agent_advisor(client_advisor, results)
print(report)

# %%

import numpy as np

# Fixed seed for reproducibility
np.random.seed(42)

# Real thesis scores: {question: score}
real_scores = {
    "Does it describe a feasible research methodology?": 8.5,
    "Does the proposal include a well-defined problem statement?": 9.0,
    "Does it include a scientifically rigorous evaluation strategy?": 8.0,
    "Does it include a model comparison?": 8.5,
    "Does it describe an analysis of errors?": 7.0,
}

# Randomized thesis scores: {question: [score_1, ..., score_1000]}
random_scores = {
    "Does it describe a feasible research methodology?": np.clip(np.random.normal(5.5, 1.0, 1000), 0, 10),
    "Does the proposal include a well-defined problem statement?": np.random.beta(3, 4, 1000) * 10,
    "Does it include a scientifically rigorous evaluation strategy?": np.clip(np.random.normal(5.0, 1.5, 1000), 0, 10),
    "Does it include a model comparison?": np.clip(np.random.gamma(5, 1, 1000), 0, 10),
    "Does it describe an analysis of errors?": np.clip(np.random.normal(5.5, 1.2, 1000), 0, 10),
}


