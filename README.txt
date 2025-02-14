All source for process data is saved in the RAG_main.typing file
* Functions in the main.ipyng file
- Initialize and save the local mode
- Process data
- Save chunk, embedding documents, and Faiss
- Test answer with API

***Please load and save the embedding model (in file RAG_main.iypnb)
and create the virtual environment before starting the demo
***

* Requiments libraries:
pip install sentence-transformers
pip install python-docx
pip install numpy
pip install faiss-cpu
pip install google-generativeai
pip install groq
pip install pickle-mixin
...

run in cmd:
(khởi tạo môi trường ảo cho streamlit)
step1: python -m venv ragenv
step2: ragenv\Scripts\activate
step3: install library (sentence-transformers, faiss-cpu, google-generativeai, groq,...)

step4: Run app.py by streamlit: streamlit run app.py
