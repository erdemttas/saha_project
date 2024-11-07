import os
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import fasttext
from deep_translator import GoogleTranslator


load_dotenv()

my_key_openai = os.getenv("my_key_openai")
model = fasttext.load_model("lid.176.bin")


def detect(message):
    detected_language = model.predict(message)[0][0].replace("__label__", "")
    prompt = f"""
    message: {message},
    if this message is outside the context of hotel, cafe, or restaurant, respond with "out of context."
    if this message is within the context of these headings, respond with "in context."
    """

    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=my_key_openai)

    response = llm.invoke(prompt)
    answer = response.content


    translated_text = GoogleTranslator(source='auto', target=detected_language).translate(answer)
    return translated_text








