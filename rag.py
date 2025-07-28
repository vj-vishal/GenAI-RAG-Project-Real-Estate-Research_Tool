from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from prompt import PROMPT, EXAMPLE_PROMPT

load_dotenv()

CHUNK_SIZE= 500
CHUNK_OVERLAP = 40
EMBEDDING_MODEL= "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME= "real_estate"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"

llm = None
vector_store= None

def initialize_compnents():
    global llm, vector_store

    if llm is None:
        llm= ChatGroq(model="llama-3.3-70b-versatile", temperature= 0.9, max_tokens= 500)

    if vector_store is None:
        ef= HuggingFaceEmbeddings(
            model_name= EMBEDDING_MODEL
        )

        vector_store= Chroma(
            collection_name= COLLECTION_NAME,
            embedding_function= ef,
            persist_directory= str(VECTORSTORE_DIR)
        )

def process_urls(urls):

    yield "Initializing Components"
    initialize_compnents()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader= UnstructuredURLLoader(urls= urls)
    data= loader.load()

    yield "Splitting text into chunks...✅"
    text_splitter= RecursiveCharacterTextSplitter(
        separators= ["\n\n","\n","."," "],
        chunk_size= CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    docs= text_splitter.split_documents(data)

    yield "Add chunks to vector database...✅"
    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")


    qa_chain= load_qa_with_sources_chain(llm, chain_type="stuff",
                                        prompt=PROMPT,
                                        document_prompt= EXAMPLE_PROMPT)
    chain= RetrievalQAWithSourcesChain(combine_documents_chain= qa_chain , retriever= vector_store.as_retriever(),
                                       reduce_k_below_max_tokens=True, max_tokens_limit= 500,
                                       return_source_documents=True)

    result= chain.invoke({"question": query}, return_only_outputs= True)
    sources_docs= [doc.metadata["source"] for doc in result["source_documents"]]

    return result["answer"], sources_docs

if __name__ == "__main__":
    urls=[
        "https://www.ndtv.com/world-news/donald-trump-jerome-powell-united-states-stock-market-news-latest-us-markets-rattled-amid-trumps-firing-federal-chief-confusion-8891822",
        "https://www.ndtvprofit.com/global-economics/us-mortgage-rates-increase-for-first-time-in-nine-weeks"
    ]

    process_urls(urls)
    answer,source= generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer: {answer}")
    print(f"Source: {source}")
