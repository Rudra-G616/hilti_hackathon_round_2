from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import StuffDocumentsChain, MapReduceDocumentsChain
from langchain.chains import LLMChain
from typing import List, Dict
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from operator import itemgetter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

#===================================================================================================#

def create_embeddings(processed_chunks: List[Dict]) -> FAISS:
    """
    Generate embeddings for chunks and store them in a FAISS vector database.
    """
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Converting chunks to Documents...")
    documents = [
        Document(
            page_content=chunk["chunk_text"],
            metadata={
                "file_path": chunk["file_path"],
                "filename": chunk["filename"],
                "page": chunk["page"]
            }
        )
        for chunk in processed_chunks
    ]

    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)

    print("FAISS index created.")
    return vectorstore

#===================================================================================================#

# def setup_hybrid_retriever(vectorstore: FAISS) -> RetrievalQA:
#     """
#     Set up a hybrid retriever combining embedding-based and keyword-based search.
#     """
#     print("Setting up hybrid retriever...")
#     retriever = vectorstore.as_retriever()

#     # Load GPT-2 model
#     llm = HuggingFaceHub(
#     repo_id="gpt2",  # Replace FLAN-T5 with GPT-2 model identifier
#     model_kwargs={"temperature": 0, "max_length": 256}
#     )

#     prompt_template = PromptTemplate(
#         input_variables=["context", "question"],
#         template="Given the following context: {context}, answer the question: {question}"
#     )

#     llm_chain = LLMChain(llm=llm, prompt=prompt_template)

#     combine_docs_chain = MapReduceDocumentsChain(
#     llm_chain=llm_chain,  # LLMChain to process documents
#     document_variable_name="document"  # Specify how the documents will be passed
#     )

#     qa_chain = RetrievalQA(
#         retriever=retriever,
#         llm_chain =llm_chain,
#         combine_documents_chain=combine_docs_chain,
#         return_source_documents=True
#     )

#     print("Hybrid retriever setup complete.")
#     return qa_chain

#===================================================================================================#
# def setup_hybrid_retriever(vectorstore: FAISS):
#     """
#     Set up a hybrid retriever combining embedding-based and keyword-based search using modern LangChain syntax.
#     """
#     print("Setting up hybrid retriever...")
    
#     # Set up the retriever
#     retriever = vectorstore.as_retriever()

#     # Initialize the LLM with updated import
#     llm = HuggingFaceEndpoint(
#         repo_id="gpt2",
#         temperature = 0,
#         max_length = 256
#     )

#     # Create prompt templates
#     map_prompt = PromptTemplate.from_template("""
#     The following is a document: {document}
#     Based on this document, provide a relevant response to: {question}
#     """)

#     combine_prompt = PromptTemplate.from_template("""
#     Given the following context: {context}
#     Answer the question: {question}
#     """)

#     # Create the document processing chain
#     map_chain = map_prompt | llm | StrOutputParser()

#     # Function to process documents
#     def process_docs(docs: List[str], question: str):
#         results = []
#         for doc in docs:
#             result = map_chain.invoke({"document": doc, "question": question})
#             results.append(result)
#         return "\n".join(results)

#     # Create the main retrieval chain
#     retrieval_chain = RunnableParallel(
#         context=retriever | process_docs,
#         question=RunnablePassthrough()
#     ) | combine_prompt | llm | StrOutputParser()

#     print("Hybrid retriever setup complete.")
#     return retrieval_chain


# def query_chain(chain, question: str):
#     """
#     Query the chain with a question and return the response
#     """
#     response = chain.invoke(question)
#     return response

#===================================================================================================#

# def setup_hybrid_retriever(vectorstore: FAISS):
#     """
#     Set up a hybrid retriever combining embedding-based and keyword-based search using modern LangChain syntax.
#     """
#     print("Setting up hybrid retriever...")
    
#     # Set up the retriever
#     retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

#     # Initialize the LLM
#     llm = HuggingFaceEndpoint(
#         repo_id="gpt2",
#         temperature=0.3,
#         max_length=256
#     )

#     # Create prompt templates
#     map_prompt = PromptTemplate.from_template("""
#     The following is a document: {document}
#     Based on this document, provide a relevant response to: {question}
#     """)

#     combine_prompt = PromptTemplate.from_template("""
#     Given the following context: {context}
#     Answer the question: {question}
#     """)

#     # Create the document processing chain
#     map_chain = map_prompt | llm | StrOutputParser()

#      # Truncate long documents
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

#     def truncate_document(doc_content: str, max_tokens: int = 500):
#         chunks = text_splitter.split_text(doc_content)
#         truncated_text = " ".join(chunks[:max_tokens // len(chunks)])
#         return truncated_text

#     # Function to process documents
#     def process_docs(input_dict: Dict):
#         docs = input_dict["documents"]
#         question = input_dict["question"]
#         results = []
#         for doc in docs:
#             truncated_content = truncate_document(doc.page_content)
#             result = map_chain.invoke({"document": truncated_content, "question": question})
#             results.append(result)
#         return "\n".join(results)

#     # Define the retrieval chain
#     retrieval_chain = RunnableParallel({
#         "documents": retriever,
#         "question": RunnablePassthrough()
#     }) | {
#         "context": process_docs,
#         "question": itemgetter("question")
#     } | combine_prompt | llm | StrOutputParser()

#     print("Hybrid retriever setup complete.")
#     return retrieval_chain

#===================================================================================================#

# class CustomHuggingFaceEndpoint(HuggingFaceEndpoint):
#     def generate_prompt(self, prompts, stop=None, callbacks=None, **kwargs):
#         # Define a set of supported kwargs for your specific model
#         supported_kwargs = {'temperature', 'max_new_tokens'}  # Add other supported kwargs as needed

#         # Filter out unsupported kwargs
#         filtered_kwargs = {k: v for k, v in kwargs.items() if k in supported_kwargs}

#         # Call the superclass method with filtered kwargs
#         return super().generate_prompt(prompts, stop=stop, callbacks=callbacks, **filtered_kwargs)

#===================================================================================================#



def setup_hybrid_retriever(vectorstore: FAISS):
    """
    Set up a hybrid retriever combining embedding-based and keyword-based search using modern LangChain syntax.
    """
    print("Setting up hybrid retriever...")

    # Set up the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize the LLM (Flan-T5)
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",  # Use Flan-T5 model
        model_kwargs = {"temperature":1, "max_new_tokens":250}  # Adjust max_new_tokens to balance input and output
    )

    print(llm)

    # Create prompt templates
    map_prompt = PromptTemplate.from_template("""
    The following is a document: {document}
    Based on this document, provide a relevant response to: {question}
    """)

    combine_prompt = PromptTemplate.from_template("""
    Given the following context: {context}
    Answer the question: {question}
    """)

    # Create the document processing chain
    map_chain = map_prompt | llm | StrOutputParser()

    # Truncate long documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    def truncate_document(doc_content: str, max_tokens: int = 500):
        chunks = text_splitter.split_text(doc_content)
        if len(chunks) == 0:
            return ""
        truncated_text = " ".join(chunks[:max_tokens // len(chunks)])
        return truncated_text

    # Function to process documents
    def process_docs(input_dict: Dict):
        docs = input_dict["documents"]
        question = input_dict["question"]
        results = []
        for doc in docs:
            truncated_content = truncate_document(doc.page_content)
            result = map_chain.invoke({"document": truncated_content, "question": question})
            results.append(result)
        return "\n".join(results)

    # Define the retrieval chain
    retrieval_chain = RunnableParallel({
        "documents": retriever,
        "question": RunnablePassthrough()
    }) | {
        "context": process_docs,
        "question": itemgetter("question")
    } | combine_prompt | llm | StrOutputParser()

    print("Hybrid retriever setup complete.")
    return retrieval_chain

#===================================================================================================#

# # Example usage
# file_path = '/mnt/data/sample_text_file.txt'  # Path to your file
# chunk_size = 100  # Customize chunk size if needed

# # Step 1: Process file to create chunks
# processed_chunks = process_file(file_path, chunk_size=chunk_size)

# # Step 2: Create embeddings and FAISS vectorstore
# vectorstore = create_embeddings(processed_chunks)

# # Step 3: Set up hybrid retriever with FLAN-T5
# qa_system = setup_hybrid_retriever(vectorstore)

# # Save FAISS index for future use
# vectorstore.save_local('/mnt/data/faiss_index')
# print("FAISS index saved.")

#===================================================================================================#

tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

def grammar_correction(text):
    input_text = text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=502)
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text