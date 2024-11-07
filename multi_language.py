import os
import fasttext
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


model = fasttext.load_model("lid.176.bin")
load_dotenv()

my_key_openai = os.getenv("my_key_openai")
llm_openai = ChatOpenAI(model="gpt-4o-mini", api_key=my_key_openai)


store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

session_id = "firstChat"
with_message_history = RunnableWithMessageHistory(llm_openai, get_session_history)




def detect_language_switch(messages):
    current_language = None
    message = messages[0]
    detected_language = model.predict(message.content)[0][0].replace("__label__", "")


    #---------------------------------------------------------------------------------------------------------------
    def load_language_codes(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return {line.split(":")[0].strip(): line.split(":")[1].strip() for line in file if ":" in line}

    language_codes = load_language_codes('languages.txt')

    detected_language = language_codes.get(detected_language, "Unknown Language")
    #-----------------------------------------------------------------------------------------------------------------


    if current_language is None:
        current_language = detected_language
    elif current_language != detected_language:
        print(f"Language switch detected: {current_language} -> {detected_language}")

    current_language = detected_language
    return current_language




previous_languages = []
def detect_and_respond(text):
    messages = [HumanMessage(content=text)]
    change = ""
    detected_language = detect_language_switch(messages)
    if detected_language and (not previous_languages or previous_languages[-1] != detected_language):
        previous_languages.append(detected_language)
        print(f"Language switch detected: {previous_languages[-2] if len(previous_languages) > 1 else 'N/A'} -> {detected_language}")
        change = f"Language switch detected: {previous_languages[-2] if len(previous_languages) > 1 else 'N/A'} -> {detected_language}"



    system_message = SystemMessage(content=f"Response in this language: {detected_language if detected_language else 'N/A'}")

    response = with_message_history.invoke(
        [system_message] + messages,
        config={"configurable": {"session_id": session_id}},
    )

    return response.content, change



























