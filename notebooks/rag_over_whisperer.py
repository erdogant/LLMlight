# Import the necessary libraries
import whisper

# Load the base model from Whisper
model = whisper.load_model("base")

# Add your Audio File
audio = r"D:\notabene\01_2025-05-07 MT IenE Heisessie - 1 - Opening.m4a"

# Transcribe the audio file
result = model.transcribe(audio, fp16=False)

result = {}
result['text'] =  "I can conceive of a national destiny which meets the responsibilities of today and measures up to the possibilities of tomorrow. Behold a republic resting securely upon the mountain of eternal truth. A republic applying in practice and proclaiming to the world the self-evident propositions that all men are created equal, that they are endowed with inalienable rights, that governments are instituted among men to secure these rights, and that governments derive their just powers from the consent of the governed. Behold a republic in which civil and religious liberty stimulate all to earnest endeavor, and in which the law restrains every hand uplifted for a neighbor's injury. A republic in which every citizen is a sovereign, but in which no one cares to wear a crown. Behold a republic standing erect while empires all around or bow beneath the weight of their own armaments. A republic whose flag is love while other flags are only fears. Behold a republic increasing in population, in wealth, in strength, and in influence, solving the problems of civilization and facing the coming of a universal brotherhood. A republic which shakes, throes, and dissolves aristocracies by its silent example and gives light and inspiration to those who sit in darkness. Behold a republic gradually but surely becoming a supreme moral factor in the world's progress and the accepted arbiter of the world's dispute. A republic whose history, like the path of the just, is as the shining light that shineth more and more unto the perfect day."

print(result["text"])

#%% Step 2: Tokenize and Embed the Text
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama


# Tokenizing and creating embeddings allow us to split the transcription into smaller chunks and find similarities between them. We use LangChain for this purpose, specifically the RecursiveCharacterTextSplitter and Ollama Embeddings.

# Define the text to split
transcription = result["text"]

# Split the text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = splitter.split_text(transcription)

# Print the texts to get an initial look
print(texts)

# Create embeddings for each chunk
embeddings = OllamaEmbeddings()

# Create the vector store using FAISS
docsearch = FAISS.from_texts(texts, embeddings, metadatas=[{"source": str(i)} for i in range(len(texts))])


#%% Step 3: Set up the Local LLM Model and Prompt

# Set up the LLM (you will need to install llama2 using Ollama)
# Now, we define the local LLM model (Ollama) and set up the prompt for the RAG system. Note: you need to download the model you’d like to use with Ollama.
# https://ollama.com/


llm = Ollama(model='llama2')
llm = Ollama(model='qwen3:4b')


#import chatprompttemplate 
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Create the RAG prompt
rag_prompt = ChatPromptTemplate(
    input_variables=['context', 'question'], 
    messages=[
        HumanMessagePromptTemplate(
            prompt=PromptTemplate(
                input_variables=['context', 'question'], 
                template="""You answer questions about the contents of a transcribed audio file. 
                Use only the provided audio file transcription as context to answer the question. 
                Do not use any additional information.
                If you don't know the answer, just say that you don't know. Do not use external knowledge. 
                Use three sentences maximum and keep the answer concise. 
                Make sure to reference your sources with quotes of the provided context as citations.
                \nQuestion: {question} \nContext: {context} \nAnswer:"""
                )
        )
    ]
)

# load in qa_chain
from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff", prompt=rag_prompt)

#%% Step 4: Set a Query and Find Similar Documents

# Define a query
query = "What are the self-evident propositions in this speech?"

# Find similar documents to the search query
docs = docsearch.similarity_search(query)
print(docs)

#%% Step 5: Generate a Response Using Chain Completion

# Set a response variable to the output of the chain
response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)

# Display the response
print("Based on the provided context, the self-evident propositions in the speech are:")
print("\n".join(response["output_text"]))

