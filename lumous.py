import streamlit as st
from datetime import date
from src.workflow import workflow
from langchain_core.messages import HumanMessage
#from src.data_ingestion import main
from src.workflow import workflow
import speech_recognition as sr
from openai import OpenAI
import base64
from PIL import Image

logo = Image.open('puffy_logo.jpg')
st.image(logo,width=150)

st.set_page_config(page_title="Puffy Lumous - Smart Search", layout="centered")


st.markdown("""
    <style>
    .search-box {
        border: 2px solid #444;
        padding: 20px;
        border-radius: 10px;
        background-color: #f9f9f9;
        margin-bottom: 20px;
    }
    .search-title {
        font-weight: bold;
        font-size: 16px;
        margin-bottom: 5px;
    }
    </style>
""", unsafe_allow_html=True)

# 1️⃣ Text Search (Press Enter)
search_query = st.text_input("**Type your search query**", placeholder="Search for Puffy products...")

# 2️⃣ Voice Input (Separate mic button)
st.markdown("**🎤 Voice Input**", unsafe_allow_html=True)
voice_query = st.button("**Speak or record your query here**")

# 3️⃣ Image Upload
st.markdown("**📷 Upload an Image**", unsafe_allow_html=True)
uploaded_image = st.file_uploader("**Choose an image...**", type=["png", "jpg", "jpeg"],
                                  help="Upload a room, bed, or sofa image to find similar Puffy products.")
    
# --- Backend logic for text search (already exists in your code)
if voice_query:
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Speak now!")
        audio = r.listen(source)

    try:
        text = r.recognize_google(audio)
        st.success(f"✅ You said: {text}")
        st.write(f"Searching for: **{text}**")
        app = workflow()
        message = [HumanMessage(content=text)]
        response = app.invoke({"user_query":message})
        print(f"*****response***** {response}")
        try:
            final_output = response["llm_response"].content
        except:
            final_output = response["messages"][-1].content
        st.success("✅ Recommendations ready!")
        print('\n')
        print(f"*****final_output***** {final_output}")
        st.write(final_output)        

        # Reuse search logic
        if text.strip():
            st.write(f"Searching for: **{text}**")
    except sr.UnknownValueError:
        st.error("❌ Could not understand audio")
    except sr.RequestError:
        st.error("❌ API unavailable")


if search_query.strip():
    st.write(f"Searching for: **{search_query}**")
    app = workflow()
    message = [HumanMessage(content=search_query)]
    response = app.invoke({"user_query":message})
    print(f"*****response***** {response}")
    final_output = response["llm_response"].content
    st.success("✅ Recommendations ready!")
    print('\n')
    print(f"*****final_output***** {final_output}")
    st.write(final_output)

if uploaded_image:
    st.caption("🛈 We'll analyze the image and suggest the closest matching Puffy products.")
    client = OpenAI()
    image_bytes = uploaded_image.read()
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    response = client.responses.create(
    model="gpt-4.1",
    input=[
        {"role": "system", "content": "You are a helpful assistant to generate a prompt based on provided image. No need of color details. focus on what the product is all about"},
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Create 2-3 line prompt to compare the provided image description with puffy product"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
            ],
        },
    ],
)

    description = response.output_text
    #st.write(description)
    app = workflow()
    message = [HumanMessage(content=description)]
    response = app.invoke({"user_query":message})
    print(f"*****response***** {response}")
    final_output = response["llm_response"].content
    st.success("✅ Recommendations ready!")
    print('\n')
    print(f"*****final_output***** {final_output}")
    st.write(final_output)


