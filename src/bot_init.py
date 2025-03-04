import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src import commonconstants

class InformationRetriever():
    def __init__(self, llm_model):
        self.llm_model = llm_model
        print(f"Initializing Bot() with llm_model: {self.llm_model}")

        if not os.path.isdir(commonconstants.DB_KNOWLEDGE_FAISS_PATH):
            print(f"Knowledge Directory does not exist. So converting the data from {commonconstants.DATA_PATH} to {commonconstants.DB_KNOWLEDGE_FAISS_PATH} (vector database)")
            self.load_data()
        self.qa_bot()

    def load_data(self):
        print("Entering load_data()")
        print("Please wait... This may take a some time to feed the data to vector space.")
        print("Note: This will run only for the first time you run this application without the knowledge directory")
        documents_loader_pdf = DirectoryLoader(
            commonconstants.DATA_PATH,
            glob="*.pdf",
            loader_cls=PyPDFLoader)
        
        documents = documents_loader_pdf.load()
        if len(documents) > 0:
            print("Detected *.pdf contents")

        documents_loader_txt = DirectoryLoader(
            commonconstants.DATA_PATH,
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
            )
        
        txt_documents_list = documents_loader_txt.load()
        if len(txt_documents_list) > 0:
            print("Detected *.txt contents")
            documents.extend(txt_documents_list)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )

        docs = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(commonconstants.DB_KNOWLEDGE_FAISS_PATH)
        print("Completed load_data()")

    def qa_bot(self):
        embeddings = HuggingFaceEmbeddings(model_name=commonconstants.EMBEDDINGS_MODEL)
        db_library = FAISS.load_local(commonconstants.DB_KNOWLEDGE_FAISS_PATH, embeddings=embeddings, allow_dangerous_deserialization=True)
        self.db_library = db_library

    def get_relavent_content(self, question: str, knowledge_level: int):
        result = self.db_library.similarity_search(query=question, k=knowledge_level)

        full_content = ''
        for content in result:
            full_content = full_content + content.page_content

        return full_content

    def set_custom_prompt_template(self, question, knowledge_level: int):
        print(f"Entering set_custom_prompt_template() for {self.llm_model}")

        CUSTOM_PROMPT_TEMPLATE = commonconstants.GEMMA_PROMPT_TEMPLATE

        CUSTOM_PROMPT_TEMPLATE_MAIN = CUSTOM_PROMPT_TEMPLATE.replace("#topic#", commonconstants.RELATED_TOPICS)
        CUSTOM_PROMPT_TEMPLATE_MAIN = CUSTOM_PROMPT_TEMPLATE_MAIN.replace("#COMPANY_NAME#", commonconstants.COMPANY_NAME)

        context = self.get_relavent_content(question, knowledge_level)
        
        prompt = CUSTOM_PROMPT_TEMPLATE_MAIN.format(context=context, question=question)
        print(f"Completed set_custom_prompt_template() for {self.llm_model}")
        return prompt
    
    def generate_prompt(self, query: str, knowledge_level: int):
        prompt = self.set_custom_prompt_template(question=query, knowledge_level=knowledge_level)
        print(prompt)
        return prompt



information_retriever = InformationRetriever(llm_model=commonconstants.LLM_MODEL_ID)