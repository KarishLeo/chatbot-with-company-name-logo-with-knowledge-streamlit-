import requests
import json
import streamlit as st
from src.bot_init import information_retriever
from src import commonconstants
from datetime import datetime

def load_css(file_path):
    with open(file_path, "r") as f:
        css_content = f.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "bot", "content": commonconstants.WELCOME_MSG.format(company_name=commonconstants.COMPANY_NAME)}
    ]

if "user_input" not in st.session_state:
    st.session_state.user_input = ""


def stream_ollama_response(prompt):
    try:
        with open(commonconstants.CONVERSATION_HISTORY_LOGGER_FILE, "a", encoding="utf-8") as file:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write("formatted_time:" + formatted_time + " " + prompt + "\n\n" + "="*50 + "\n\n")


        payload = {
            "model": commonconstants.LLM_MODEL_ID,
            "prompt": prompt,
            "stream": True
        }

        response_text = ""

        with requests.post(commonconstants.OLLAMA_API_URL, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            response_data = json.loads(line.decode("utf-8"))
                            chunk = response_data.get("response", "")
                            response_text += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                yield f"Error: {response.status_code} - {response.text}"

        with open(commonconstants.CONVERSATION_HISTORY_LOGGER_FILE, "a", encoding="utf-8") as file:
            file.write("Assistant Response:\n" + response_text + "\n\n" + "="*50 + "\n\n")
    except Exception as e:
        return str(e)


def stream_ollama_response_with_conversation_history(prompt):
    try:
        conversation_history = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages[-3:]]
        )
        
        full_prompt = f"Conversation history:\n{conversation_history}\n\nUser: {prompt}\nAssistant:"

        with open(commonconstants.CONVERSATION_HISTORY_LOGGER_FILE, "a", encoding="utf-8") as file:
            now = datetime.now()
            formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
            file.write("formatted_time:" + formatted_time + " " + full_prompt + "\n\n" + "="*50 + "\n\n")

        payload = {
            "model": commonconstants.LLM_MODEL_ID,
            "prompt": full_prompt,
            "stream": True
        }

        response_text = ""

        with requests.post(commonconstants.OLLAMA_API_URL, json=payload, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            response_data = json.loads(line.decode("utf-8"))
                            chunk = response_data.get("response", "")
                            response_text += chunk
                            yield chunk
                        except json.JSONDecodeError:
                            continue
            else:
                error_message = f"Error: {response.status_code} - {response.text}"
                yield error_message
                response_text = error_message

        with open(commonconstants.CONVERSATION_HISTORY_LOGGER_FILE, "a", encoding="utf-8") as file:
            file.write("Assistant Response:\n" + response_text + "\n\n" + "="*50 + "\n\n")
    except Exception as e:
        return str(e)


def main():
    st.set_page_config(page_title=f"{commonconstants.COMPANY_NAME} Intelligence", layout="wide")
    
    load_css("resources/styles.css")

    col1, col2 = st.columns([1, 8])
    with col1:
        st.image(commonconstants.COMPANY_LOGO_FILE, width=80)
    with col2:
        st.title(f"{commonconstants.COMPANY_NAME} Intelligence (ChatBot)")
        st.text(commonconstants.CHATBOT_DESCRIPTION)

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            role = message["role"]
            content = message["content"]

            if role == "user":
                st.markdown(f'<div class="user-message">🧑 {content}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-message">🤖 {content}</div>', unsafe_allow_html=True)

    st.markdown('<div class="input-container">', unsafe_allow_html=True)

    query_input = st.text_area(
        "Enter your prompt:",
        placeholder="Type your query here...",
        key="user_input"
    )

    send_button = st.button("Send")
    st.markdown('</div>', unsafe_allow_html=True)

    if send_button:
        req_str = query_input
        del st.session_state["user_input"]
        if req_str.strip():
            context = {"role": "user", "content": req_str}
            print(f"--> Content: {context}")
            st.session_state.messages.append(context)

            chat_container.markdown(f'<div class="user-message">🧑 {req_str}</div>', unsafe_allow_html=True)

            bot_message_placeholder = chat_container.empty()
            full_response = ""

            custom_prompt_str = information_retriever.generate_prompt(query=req_str,
                    knowledge_level=commonconstants.KNOWLEDGE_LEVEL)

            print(">>>>>>>>>>>>>>>SENTING REQUEST TO LLM>>>>>>>>>>>>>>>")
            if commonconstants.CHAT_HISTORY_FEED_TO_LLM_FLAG:
                for chunk in stream_ollama_response_with_conversation_history(custom_prompt_str):
                    if chunk.lower() == "gemma":
                        chunk = f"{commonconstants.COMPANY_NAME} Intelligence"
                    full_response += chunk
                    bot_message_placeholder.markdown(f'<div class="bot-message">🤖 {full_response}</div>', unsafe_allow_html=True)
            else:
                for chunk in stream_ollama_response(custom_prompt_str):
                    if chunk.lower() == "gemma":
                        chunk = f"{commonconstants.COMPANY_NAME} Intelligence"
                    full_response += chunk
                    bot_message_placeholder.markdown(f'<div class="bot-message">🤖 {full_response}</div>', unsafe_allow_html=True)

            st.session_state.messages.append({"role": "bot", "content": full_response})

            if hasattr(st, "rerun"):
                st.rerun()
            else:
                st.experimental_rerun()

        else:
            st.warning("Please enter a prompt.")


if __name__ == "__main__":
    main()