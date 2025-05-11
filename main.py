import os
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain.schema import SystemMessage
from datasets import load_dataset
from pinecone import Pinecone, ServerlessSpec, CloudProvider, AwsRegion, Metric
from langchain_pinecone import PineconeVectorStore
from tqdm.auto import tqdm

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

# Load and convert dataset
dataset = load_dataset("jamescalam/deepseek-r1-paper-chunked", split="train")
data = dataset.to_pandas()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "rag-chatbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        metric=Metric.DOTPRODUCT,
        dimension=1536,
        spec=ServerlessSpec(cloud=CloudProvider.AWS, region=AwsRegion.US_EAST_1)
    )

index = pc.Index(name=index_name)

# Initialize embeddings
embed_model = CohereEmbeddings(
    cohere_api_key=COHERE_API_KEY,
    model="embed-v4.0"
)

# Upload data to Pinecone
batch_size = 100
for i in tqdm(range(0, len(data), batch_size)):
    batch = data.iloc[i:i + batch_size]
    ids = [f"{x['doi']}-{x['chunk-id']}" for _, x in batch.iterrows()]
    texts = [x['chunk'] for _, x in batch.iterrows()]
    embeds = embed_model.embed_documents(texts)
    metadata = [{"text": x['chunk'], "source": x['source']} for _, x in batch.iterrows()]
    index.upsert(vectors=zip(ids, embeds, metadata))

# Initialize vector store
vectorstore = PineconeVectorStore(
    index=index,
    embedding=embed_model,
    text_key="text"
)

# Initialize ChatCohere
chat = ChatCohere(
    cohere_api_key=COHERE_API_KEY,
    model="command-r",
    temperature=0.5
)

# Function to build augmented prompt
def augment_prompt(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    if not results:
        return "No context found. Please try a different question."
    source_knowledge = "\n".join([x.page_content for x in results])
    return f"""You are a helpful research assistant. Answer using only the context below. If unsure, say "I don't know based on the provided context."

Contexts:
{source_knowledge}

Query: {query}"""


if __name__ == "__main__":
    # Interactive loop
    print("🔍 Ask anything about Deepseek R1 (type 'exit' to quit):")
    while True:
        user_query = input("\n🧠 Your question (type 'exit' to quit): ")
        if user_query.strip().lower() in ("exit", "quit"):
            break

        # Reset conversation messages
        messages = [
            SystemMessage(content=(
                "You are a helpful assistant who answers using only the provided context."
            )),
            HumanMessage(content=augment_prompt(user_query))
        ]

        response = chat.invoke(messages)
        print("\n🤖 Answer:\n", response.content)

    # Optionally delete index
    # pc.delete_index(index_name)