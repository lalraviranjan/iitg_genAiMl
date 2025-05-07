import re
import os
from dotenv import load_dotenv
import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.docstore.document import Document


load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")

## sstreamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

def load_youtube_transcript(video_url):
    # Extract video ID from URL
    video_id_match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not video_id_match:
        raise ValueError("Invalid YouTube URL")
    video_id = video_id_match.group(1)

    # Fetch transcript
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
    text = " ".join([entry["text"] for entry in transcript])
    return [Document(page_content=text)]

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma2-9b-It")

generic_url=st.text_input("URL",label_visibility="collapsed")

prompt_template="""
Provide a summary of the following content in not more than 2000 words:
Keep the summary in pointers and make sure to include the important points from the content.
Use different meaningful icons for each pointers.
At the end include in 1-2 lines what other things relatively to the topic can be explored not mentioned in the content.
Content:{text}

"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the information to get started")
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can may be a YT video url or website url")

    else:
        try:
            with st.spinner("Waiting..."):
                ## loading the website or yt video data
                if "youtube.com" in generic_url:
                    docs=load_youtube_transcript(generic_url)
                else:
                    loader=UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )
                    docs=loader.load()

                ## Chain For Summarization
                chain=load_summarize_chain(llm,chain_type="stuff",prompt=prompt)
                output_summary=chain.run(docs)

                st.markdown(output_summary)

        except Exception as e:
            st.exception(f"Exception:{e}")