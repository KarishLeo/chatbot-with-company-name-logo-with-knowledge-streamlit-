# Enter your company name and logo path
COMPANY_NAME: str = "CompanyName"
CHATBOT_DESCRIPTION: str = "Description about your chatbot"
COMPANY_LOGO_FILE: str = "resources/icons/company_logo.png"

# Related topics to your domain
RELATED_TOPICS: str = 'Global warming'

OLLAMA_API_URL: str = "http://localhost:11434/api/generate"

# Data path
DATA_PATH: str = 'resources/data'
DB_KNOWLEDGE_FAISS_PATH: str = 'resources/vectorstore/stored_knowledge'
CONVERSATION_HISTORY_LOGGER_FILE: str = 'resources/conversation_log.txt'

EMBEDDINGS_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_LEVEL: int = 25 # This level indicates the number of documents to be retrived by the similarity search algorithm

#LLM MODEL CHOICE (MAKE SURE YOU HAVE PULL THESE MODELS IN OLLAMA USING 'ollama pull [llm_model]')
GEMMA2_llm_model: str = "gemma2:9b"            
LLAMA3_llm_model: str = "llama3.1:8b"          
LLAMA3_CHATQA_llm_model: str = "llama3-chatqa:8b"     
DEEPSEEK_R1_5B_llm_model: str = "deepseek-r1:1.5b"     
DEEPSEEK_R1_7B_llm_model: str = "deepseek-r1:7b"
PHI3_MODEL3_8B_llm_model: str = "phi3:3.8b"
ORCA2_7B_llm_model: str = "orca2:7b"
MISTRAL_LATEST_llm_model: str = "mistral-openorca:7b"

# LLM MODEL TO RUN THE APPLICATION
LLM_MODEL_ID: str = GEMMA2_llm_model

# To enable conversational history
CHAT_HISTORY_FEED_TO_LLM_FLAG: bool = False

WELCOME_MSG: str = """
Hi, I am {company_name} Intelligence. 🤖\n
Your Trusted Intelligent Assistant. \n
If you're looking for the right products, need in-depth explanations, or require tailored solutions for your challenges, I'm here to assist you. \n

Share your requirements, I’ll guide you to the best technologies and innovations our company has to offer. \n
Feel free to ask me!
"""

GEMMA_PROMPT_TEMPLATE: str = """
You are #COMPANY_NAME# AI assistant specialized in answering the users query based on the provided context.  

context:  
{context}  

Question:  
{question}  

Note: 
    Don't mention based on the provided content. Just speak naturally.
    If the answer is not found in the context, respond with: "I don’t have information on that."  
    If the question is unrelated to #topic#, say: "This question is out of scope."
"""
