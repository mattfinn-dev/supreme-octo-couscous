import feedparser
import chromadb
import os
import chromadb.utils.embedding_functions as embedding_functions
from openai import AzureOpenAI
    
# Embedding function
azure_open_ai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_type="azure",
    api_version=os.getenv("OPENAI_API_VERSION"),
    model_name="text-embedding-3-large"
)

# Azure OpenAI API Client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
    api_version=os.getenv("OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

feed:feedparser.FeedParserDict = feedparser.parse("https://feeds.feedburner.com/fibre2fashion")

# print(feed.entries[0].keys())

# for entry in feed.entries:
#     print(entry.summary)

chroma_client = chromadb.PersistentClient(path="./chroma")

fashion_collection = chroma_client.get_or_create_collection(name="fibrefashion", embedding_function=azure_open_ai_ef)

fashion_collection.upsert(ids=[entry.id for entry in feed.entries],
                          metadatas=[{"title": entry.title, "url": entry.link, "published": entry.published} for entry in feed.entries],
                          documents=[entry.summary for entry in feed.entries])

def query_chromadb(prompt:str) -> str:

    results = fashion_collection.query(query_texts=prompt,
                                    n_results=5)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": f"I want to make the best {prompt} possible and you are an expert in fashion, design, marketing and communication. Using this context: {results["documents"]}, please give me advices/guidelines on what materials, colors or trends I should follow to make the fanciest and as marketable as possible apparel. Try to make your anwser quite short so i'm not bloated with informations.",
            }
        ],
        model="gpt-4",
        temperature=0.7,
        max_tokens=400
    )

    return(chat_completion.choices[0].message.content)