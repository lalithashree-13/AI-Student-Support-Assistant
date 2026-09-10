
import os
import shutil
import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from google import genai
from google.genai import types


# =========================================================
# Gemini Embeddings
# =========================================================

class NativeGeminiEmbeddings(Embeddings):

    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-embedding-001"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        return [
            embedding.values
            for embedding in response.embeddings
        ]

    def embed_query(self, text: str) -> list[float]:

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY"
            )
        )

        return response.embeddings[0].values


# =========================================================
# Streamlit Page
# =========================================================

st.set_page_config(
    page_title="AI Student Support Assistant",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 AI Student Support Assistant")

st.caption(
    "RAG-Based Student Information & Support System"
)


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    api_key = st.text_input(
        "Enter Gemini API Key",
        type="password"
    )

    st.header("🎛️ Controls")

    # -----------------------------------------------------
    # Create Knowledge Base
    # -----------------------------------------------------

    if st.button(
        "📚 Create Knowledge Base",
        use_container_width=True
    ):

        if not api_key:

            st.error(
                "Please enter your Gemini API Key first!"
            )

        elif not os.path.exists(
            "data/knowledge_base.txt"
        ):

            st.error(
                "data/knowledge_base.txt file missing!"
            )

        else:

            try:

                # Load knowledge base
                loader = TextLoader(
                    "data/knowledge_base.txt",
                    encoding="utf-8"
                )

                docs = loader.load()

                if not docs:

                    st.error(
                        "Knowledge base is empty!"
                    )

                elif not docs[0].page_content.strip():

                    st.error(
                        "knowledge_base.txt file-la content edhum illai!"
                    )

                else:

                    # Remove old database
                    if os.path.exists("chroma_db"):
                        shutil.rmtree("chroma_db")

                    # Create embeddings
                    embeddings = NativeGeminiEmbeddings(
                        api_key=api_key
                    )

                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=50
                    )

                    splits = text_splitter.split_documents(
                        docs
                    )

                    # Create Chroma database
                    Chroma.from_documents(
                        documents=splits,
                        embedding=embeddings,
                        persist_directory="chroma_db"
                    )

                    st.success(
                        "✅ Knowledge Base created successfully!"
                    )

                    st.info(
                        f"{len(splits)} knowledge chunks created."
                    )

            except Exception as e:

                st.error(
                    f"Knowledge Base Error: {e}"
                )

    # -----------------------------------------------------
    # Clear Chat
    # -----------------------------------------------------

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()

    # -----------------------------------------------------
    # Support Categories
    # -----------------------------------------------------

    st.subheader("📚 Available Support")

    st.markdown(
        """
        - 🏫 College Regulations
        - 📅 Attendance
        - 📝 Examinations
        - 📖 Course Syllabus
        - ❓ FAQs
        """
    )


# =========================================================
# Chat History
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# Chat Input
# =========================================================

question = st.chat_input(
    "Ask questions about college rules, attendance, syllabus..."
)


if question:

    # Show user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    # =====================================================
    # Check API Key
    # =====================================================

    if not api_key:

        answer = (
            "⚠️ Please enter your Gemini API Key "
            "in the sidebar."
        )

    # =====================================================
    # Check Knowledge Base
    # =====================================================

    elif not os.path.exists("chroma_db"):

        answer = (
            "⚠️ Knowledge Base has not been created yet. "
            "Please click 'Create Knowledge Base' "
            "in the sidebar first."
        )

    else:

        try:

            # Create embeddings
            embeddings = NativeGeminiEmbeddings(
                api_key=api_key
            )

            # Load Chroma database
            vectorstore = Chroma(
                persist_directory="chroma_db",
                embedding_function=embeddings
            )

            # Retrieve relevant documents
            docs = vectorstore.similarity_search(
                question,
                k=3
            )

            # Create context
            context = "\n\n".join(
                [
                    doc.page_content
                    for doc in docs
                ]
            )

            # =================================================
            # Prompt
            # =================================================

            prompt_template = ChatPromptTemplate.from_template(
                """
You are an AI Student Support Assistant.

Answer the student's question using ONLY
the information provided in the context.

If the answer is not available in the context,
politely tell the student to contact the
appropriate faculty, class advisor, or
department office.

Keep the answer clear, short and helpful.

Context:
{context}

Student Question:
{question}

Answer:
"""
            )

            formatted_prompt = prompt_template.format(
                context=context,
                question=question
            )

            # =================================================
            # Gemini LLM
            # =================================================

            llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=api_key
)

            response = llm.invoke(
                formatted_prompt
            )

            if isinstance(response.content, str):
                answer = response.content
            else:
                answer = "".join(item.get("text", "")
                for item in response.content
                if isinstance(item, dict) and item.get("type") == "text"
    )
        except Exception as e:
            
                answer = (
                "❌ Error occurred while generating "
                f"the answer:\n\n{e}"
            )


    # =====================================================
    # Display Assistant Answer
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    with st.chat_message("assistant"):

        st.markdown(answer)

