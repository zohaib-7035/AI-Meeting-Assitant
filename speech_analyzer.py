import torch
import os
import gradio as gr
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
from ibm_watsonx_ai.foundation_models.schema import TextChatParameters
from ibm_watsonx_ai.foundation_models import ModelInference
from langchain_ibm import WatsonxLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline  # For Speech-to-Text

#######------------- LLM Initialization-------------#######

# IBM Watsonx LLM credentials
project_id = "skills-network"

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
    api_key="<YOUR_API_KEY>"  # Replace with your real API key
)
client = APIClient(credentials)
model_id = "ibm/granite-3-3-8b-instruct"

parameters = {
    GenParams.DECODING_METHOD: "sample",
    GenParams.MAX_NEW_TOKENS: 512,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.TEMPERATURE: 0.5,
    GenParams.TOP_K: 50,
    GenParams.TOP_P: 1,
}

# Initialize IBM Watsonx LLM
llm = WatsonxLLM(
    model_id=model_id,
    url="https://us-south.ml.cloud.ibm.com",
    project_id=project_id,
    params=parameters,
    credentials=credentials
)

#######------------- Helper Functions-------------#######

def remove_non_ascii(text):
    return ''.join(i for i in text if ord(i) < 128)

def product_assistant(ascii_transcript):
    system_prompt = """You are an intelligent assistant specializing in financial products;
    your task is to process transcripts of earnings calls, ensuring that all references to
    financial products and common financial terms are in the correct format. For each
    financial product or common term that is typically abbreviated as an acronym, the full term 
    should be spelled out followed by the acronym in parentheses. For example, '401k' should be
    transformed to '401(k) retirement savings plan', 'HSA' should be transformed to 'Health Savings Account (HSA)', etc.
    """

    prompt_input = system_prompt + "\n" + ascii_transcript

    messages = [{"role": "user", "content": prompt_input}]

    llama = ModelInference(
        model_id="meta-llama/llama-3-2-11b-vision-instruct",
        credentials=credentials,
        project_id=project_id,
        params=TextChatParameters(temperature=0.2, top_p=0.6)
    )

    response = llama.chat(messages=messages)
    return response['choices'][0]['message']['content']

#######------------- Prompt Template and Chain-------------#######

template = """
Generate meeting minutes and a list of tasks based on the provided context.

Context:
{context}

Meeting Minutes:
- Key points discussed
- Decisions made

Task List:
- Actionable items with assignees and deadlines
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context": RunnablePassthrough()} |
    prompt |
    llm |
    StrOutputParser()
)

#######------------- Speech2text and Pipeline-------------#######

def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
    )
    raw_transcript = pipe(audio_file, batch_size=8)["text"]
    ascii_transcript = remove_non_ascii(raw_transcript)

    adjusted_transcript = product_assistant(ascii_transcript)
    result = chain.invoke({"context": adjusted_transcript})

    output_file = "meeting_minutes_and_tasks.txt"
    with open(output_file, "w") as file:
        file.write(result)

    return result, output_file

#######------------- Gradio Interface-------------#######

audio_input = gr.Audio(sources="upload", type="filepath", label="Upload your audio file")
output_text = gr.Textbox(label="Meeting Minutes and Tasks")
download_file = gr.File(label="Download the Generated Meeting Minutes and Tasks")

iface = gr.Interface(
    fn=transcript_audio,
    inputs=audio_input,
    outputs=[output_text, download_file],
    title="AI Meeting Assistant (IBM Watsonx Edition)",
    description="Upload a meeting audio file. This tool transcribes, cleans, and generates minutes + action items using IBM Watsonx LLM."
)

iface.launch(server_name="0.0.0.0", server_port=5000)
