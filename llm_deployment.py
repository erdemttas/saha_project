import streamlit as st
from multi_language import detect_and_respond
from context_analysis import detect
import time

st.set_page_config(page_title="Sohbet")
st.title("Chat with GPT 🤖💬")
st.divider()




if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(placeholder="Mesajınızı yazınız"):
    st.chat_message("user").write(prompt)

    st.session_state.messages.append({"role":"user", "content":prompt})

    with st.chat_message("assistant"):
        trail1, trail2 = detect_and_respond(prompt)


        def show_temporary_alert(trail2, prompt):
            time.sleep(0.5)
            if trail2:
                st.toast(trail2)
                time.sleep(3)
            st.toast(detect(prompt))
            time.sleep(3)


        AI_Response = trail1
        st.success(AI_Response)
        st.session_state.messages.append({"role": "assistant", "content": AI_Response})

        show_temporary_alert(trail2, prompt)













