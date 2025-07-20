

# 🤖 AI Meeting Assistant (IBM Watsonx Edition)

This Gradio-based app allows you to:

* Upload a meeting audio file.
* Automatically transcribe speech using Whisper.
* Enhance financial terminology with domain-specific context.
* Generate a structured summary and action items using **IBM Watsonx LLM** (`granite-3-3-8b-instruct`).
* Download the meeting minutes as a `.txt` file.

---

## 🚀 Features

* 🔊 Audio Transcription with `openai/whisper-medium`
* 🧠 Text refinement using financial-domain instructions
* 💼 Summarization + Task Extraction with IBM Watsonx LLM
* 🌐 Simple web interface powered by Gradio

---

## 🧰 Requirements

### Install the necessary packages:

```bash
pip install gradio torch transformers langchain langchain_ibm ibm-watsonx-ai
```

---

## 🔐 IBM Watsonx Setup

Before running the app, make sure you:

1. Have access to [IBM Watsonx.ai](https://dataplatform.cloud.ibm.com).
2. Get your **API key** and **project ID**.
3. Replace this line in the script:

```python
api_key="<YOUR_API_KEY>"  # Replace with your real API key
```

---

## 🏁 How to Run

```bash
speech_analyzer.py
```

By default, the app will be hosted locally at:

```
http://0.0.0.0:5000
```

---

## 📦 Project Structure



---

## 📋 Output Example

After uploading an audio file, the app will return:

* ✅ A detailed summary of your meeting.
* ✅ A list of action items.
* ✅ A downloadable `.txt` file with the results.

---

## 💡 Use Cases

* Financial earnings call reviews
* Corporate meeting digests
* Automated task tracking from verbal discussions

---

## 🤝 Contributing

Contributions, bug fixes, and ideas are welcome! Just open an issue or pull request.

---

## 📜 License

This project is licensed under the MIT License.

