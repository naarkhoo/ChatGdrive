"""This is a public module. It should have a docstring."""
import itertools
import os
import random
from typing import Any, List, Tuple

import streamlit as st
from langchain.agents import AgentExecutor, OpenAIFunctionsAgent
from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import QAGenerationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

st.set_page_config(page_title="PDF QA", page_icon="📚")

starter_message = "Ask me anything about the Doc!"


@st.cache_resource
def create_prompt(openai_api_key: str) -> Tuple[SystemMessage, ChatOpenAI]:
    """Create prompt."""
    # Make your OpenAI API request here
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        openai_api_key=openai_api_key,
    )

    message = SystemMessage(
        content=(
            "You are a helpful chatbot who is tasked with answering questions about context given through uploaded documents."  # noqa: E501 comment
            "Unless otherwise explicitly stated, it is probably fair to assume that questions are about the context given."  # noqa: E501 comment
            "If there is any ambiguity, you probably assume they are about that."  # noqa: E501 comment
        )
    )

    prompt = OpenAIFunctionsAgent.create_prompt(
        system_message=message,
        extra_prompt_messages=[MessagesPlaceholder(variable_name="history")],
    )

    return prompt, llm


@st.cache_data
def save_file_locally(file: Any) -> str:
    """Save uploaded files locally."""
    doc_path = os.path.join("tempdir", file.name)
    with open(doc_path, "wb") as f:
        f.write(file.getbuffer())

    return doc_path


@st.cache_data
def load_docs(files: List[Any], url: bool = False) -> str:
    """Load and process the uploaded PDF files."""
    if not url:
        st.info("`Reading doc ...`")
        documents = []
        for file in files:
            doc_path = save_file_locally(file)
            pages = PyPDFLoader(doc_path)
            documents.extend(pages.load())

    return ",".join([doc.page_content for doc in documents])


@st.cache_data
def gen_embeddings() -> HuggingFaceEmbeddings:
    """Generate embeddings for given model."""
    embeddings = HuggingFaceEmbeddings(
        cache_folder="hf_model"
    )  # https://github.com/UKPLab/sentence-transformers/issues/1828
    return embeddings


@st.cache_resource
def process_corpus(corpus: str, chunk_size: int = 1000, overlap: int = 50) -> List:
    """Process text for Semantic Search."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap
    )

    texts = text_splitter.split_text(corpus)

    # Display the number of text chunks
    num_chunks = len(texts)
    st.write(f"Number of text chunks: {num_chunks}")

    # select embedding model
    embeddings = gen_embeddings()

    # create vectorstore
    vectorstore = FAISS.from_texts(texts, embeddings).as_retriever(
        search_kwargs={"k": 4}
    )

    # create retriever tool
    tool = create_retriever_tool(
        vectorstore,
        "search_docs",
        "Searches and returns documents using the context provided as a source, relevant to the user input question.",  # noqa: E501 comment
    )

    tools = [tool]
    return tools


@st.cache_data
def generate_agent_executer(text: str) -> List[AgentExecutor]:
    """Generate the memory functionality."""
    tools = process_corpus(text)

    agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
    # Synthwave

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
    )
    return agent_executor


@st.cache_data
def generate_eval(raw_text: str, N: int, chunk: int) -> List:
    """Generate the focusing functionality."""
    # Generate N questions from context of chunk chars
    # IN: text, N questions, chunk size to draw question from in the doc
    # OUT: eval set as JSON list
    # raw_text = ','.join(raw_text)
    update = st.empty()
    ques_update = st.empty()
    update.info("`Generating sample questions ...`")
    n = len(raw_text)
    starting_indices = [random.randint(0, n - chunk) for _ in range(N)]
    sub_sequences = [raw_text[i : i + chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(llm)
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
            ques_update.info(f"Creating Question: {i+1}")
        except ValueError:
            st.warning(f"Error in generating Question: {i+1}...", icon="⚠️")
            continue

    eval_set_full = list(itertools.chain.from_iterable(eval_set))

    update.empty()
    ques_update.empty()

    return eval_set_full


@st.cache_resource()
def gen_side_bar_qa(text: str) -> None:
    """Generate responses from query."""
    if text:
        # Check if there are no generated question-answer pairs in the session state
        if "eval_set" not in st.session_state:
            # Use the generate_eval function to generate question-answer pairs
            num_eval_questions = 5  # Number of question-answer pairs to generate
            st.session_state.eval_set = generate_eval(text, num_eval_questions, 3000)

        # Display the question-answer pairs in the sidebar with smaller text
        for i, qa_pair in enumerate(st.session_state.eval_set):
            st.sidebar.markdown(
                f"""
                <div class="css-card">
                <span class="card-tag">Question {i + 1}</span>
                    <p style="font-size: 12px;">{qa_pair['question']}</p>
                    <p style="font-size: 12px;">{qa_pair['answer']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.write("Ready to answer your questions.")


# Add custom CSS
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;
    # }
        footer {visibility: hidden;
        }
        .css-card {
            border-radius: 0px;
            padding: 30px 10px 10px 10px;
            background-color: black;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 10px;
            font-family: "IBM Plex Sans", sans-serif;
        }
        .card-tag {
            border-radius: 0px;
            padding: 1px 5px 1px 5px;
            margin-bottom: 10px;
            position: absolute;
            left: 0px;
            top: 0px;
            font-size: 0.6rem;
            font-family: "IBM Plex Sans", sans-serif;
            color: white;
            background-color: green;
            }
        .css-zt5igj {left:0;
        }
        span.css-10trblm {margin-left:0;
        }
        div.css-1kyxreq {margin-top: -40px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.write(
    """
<div style="display: flex; align-items: center; margin-left: 0;">
    <h1 style="display: inline-block;">PDF GPT</h1>
    <sup style="margin-left:5px;font-size:small; color: green;">beta</sup>
</div>
""",
    unsafe_allow_html=True,
)

# Build sidebar
with st.sidebar:
    openai_api_key = st.text_input(
        "OpenAI API Key", key="api_key_openai", type="password"
    )
    if openai_api_key and openai_api_key.startswith("sk-"):
        prompt, llm = create_prompt(openai_api_key)
        memory = AgentTokenBufferMemory(llm=llm)
        "[here OpenAI API key](https://platform.openai.com/account/api-keys)"
    else:
        st.info("Please add your correct OpenAI API key in the sidebar.")

# If there's no OpenAI API key, show a message and stop the app for rendering further
if not openai_api_key:
    st.info("Please add your OpenAI API key in the sidebar.")
    st.stop()

# Use RecursiveCharacterTextSplitter as the default and only text splitter
splitter_type = "RecursiveCharacterTextSplitter"

uploaded_files = st.file_uploader(
    "Upload a PDF Document", type=["pdf"], accept_multiple_files=True
)

if uploaded_files:
    # Check if last_uploaded_files is not in session_state or
    # if uploaded_files are different from last_uploaded_files
    if (
        "last_uploaded_files" not in st.session_state
        or st.session_state.last_uploaded_files != uploaded_files
    ):
        st.session_state.last_uploaded_files = uploaded_files
        if "eval_set" in st.session_state:
            del st.session_state["eval_set"]

    # Load and process the uploaded PDF or TXT files.
    raw_pdf_text = load_docs(uploaded_files)
    st.success("Documents uploaded and processed.")

    # # Question and answering
    # user_question = st.text_input("Enter your question:")

    # embeddings = gen_embeddings()
    # gen_side_bar_qa(raw_pdf_text)

    # memory, agent_executor = generate_memory_agent_executre(raw_pdf_text)
    agent_executor = generate_agent_executer(raw_pdf_text)

if "messages" not in st.session_state or st.sidebar.button("Clear message history"):
    st.session_state["messages"] = [AIMessage(content=starter_message)]

for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    memory.chat_memory.add_message(msg)

if user_question := st.chat_input(placeholder=starter_message):
    st.chat_message("user").write(user_question)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(
            st.container(),
            expand_new_thoughts=True,
            collapse_completed_thoughts=True,
            thought_labeler=None,
        )

        response = agent_executor(
            {"input": user_question, "history": st.session_state.messages},
            callbacks=[st_callback],
            include_run_info=True,
        )
        st.session_state.messages.append(AIMessage(content=response["output"]))

        st.write(response["output"])

        memory.save_context({"input": user_question}, response)

        st.session_state["messages"] = memory.buffer

        run_id = response["__run"].run_id

        col_blank, col_text, col1, col2 = st.columns([10, 2, 1, 1])
