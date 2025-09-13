import streamlit as st
from agno.agent import Agent
from agno.knowledge.pdf import PDFKnowledgeBase, PDFReader
from agno.vectordb.qdrant import Qdrant
from agno.models.ollama import Ollama
from agno.embedder.ollama import OllamaEmbedder
from qdrant_client import QdrantClient

import tempfile
import os

#%% Define API-based Agent
class API_LLM:
    """ The Agent class.
    1. Go the LM-studio.
    2. Go to left panel and select developers mode
    3. On top select your model of interest
    4. Then go to settings in the top bar
    5. Enable "server on local network" if you need
    6. Enable Running

    Examples
    --------
    > model=API_LLM()
    > model.run('hello who are you?')

    """
    def __init__(self, model_id="openhermes-2.5-mistral-7b", api_url="http://localhost:1234/v1/chat/completions"):
        self.model_id = model_id
        self.api_url = api_url
    
    def run(self, prompt):
        import requests
        
        headers = {"Content-Type": "application/json"}
        data = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json().get('choices', [{}])[0].get('message', {}).get('content', "No response")
        else:
            return f"Error: {response.status_code} - {response.text}"


#%%
# Initialize session state
# def init_session_state():
#     if "api_agent" not in st.session_state:
#         st.session_state.api_agent = API_LLM()

#%%
def init_session_state():
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    if 'legal_team' not in st.session_state:
        st.session_state.legal_team = None
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None

def init_qdrant():
    """Initialize local Qdrant vector database with in-memory storage"""
    # Initialize Qdrant with in-memory storage
    client = Qdrant(collection="legal_knowledge", url=":memory:", embedder=OllamaEmbedder("openhermes"))
    st.write('loading qdrant client')
    st.write(client)
    return client

def process_document(uploaded_file, vector_db: Qdrant):
    """Process document using local resources"""
    if isinstance(uploaded_file, str):
        filename = os.path.basename(uploaded_file)
        # Use tempfile to create a temporary directory and save the file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, filename)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        # Open the specified file and write its content to the temporary file
        with open(uploaded_file, "rb") as source_file:
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(source_file.read())

        # try:
        st.write("Processing document...")
        # Create knowledge base with local embedder
        knowledge_base = PDFKnowledgeBase(
            path=temp_dir,
            vector_db=vector_db,
            reader=PDFReader(chunk=True),
            recreate_vector_db=True
        )

        st.write("Loading knowledge base...")
        knowledge_base.load()

        # Verify knowledge base
        st.write("Verifying knowledge base...")
        test_results = knowledge_base.search("test")
        if not test_results:
            raise Exception("Knowledge base verification failed")

        st.write("Knowledge base ready!")
        return knowledge_base

        # except Exception as e:
        #     raise Exception(f"Error processing document: {str(e)}")

def main():
    st.set_page_config(page_title="Local Legal Document Analyzer", layout="wide")
    init_session_state()

    st.title("Local AI Legal Agent Team")

    # Initialize local Qdrant
    if "vector_db" not in st.session_state:
        # try:
        st.session_state.vector_db = init_qdrant()
        st.success("Connected to local Qdrant!")
        
        st.info(st.session_state.vector_db)
        # except Exception as e:
        #     st.error(f"Failed to connect to Qdrant: {str(e)}")
        #     return

    # Document upload section
    st.header("📄 Document Upload")
    # uploaded_file = st.file_uploader("Upload Legal Document", type=['pdf'])
    uploaded_file = r'D:\\OneDrive - Tilburg University\\FAA Pilots Handbook Night Operations.pdf'
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                knowledge_base = process_document(uploaded_file, st.session_state.vector_db)
                st.write(knowledge_base)
                st.session_state.knowledge_base = knowledge_base
                
                # Initialize agents with Llama model
                legal_researcher = Agent(
                    name="Legal Researcher",
                    role="Legal research specialist",
                    model=Ollama(id="llama3.1:8b"),
                    knowledge=st.session_state.knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Find and cite relevant legal cases and precedents",
                        "Provide detailed research summaries with sources",
                        "Reference specific sections from the uploaded document"
                    ],
                    markdown=True
                )

                contract_analyst = Agent(
                    name="Contract Analyst",
                    role="Contract analysis specialist",
                    model=Ollama(id="llama3.1:8b"),
                    knowledge=knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Review contracts thoroughly",
                        "Identify key terms and potential issues",
                        "Reference specific clauses from the document"
                    ],
                    markdown=True
                )

                legal_strategist = Agent(
                    name="Legal Strategist", 
                    role="Legal strategy specialist",
                    model=Ollama(id="llama3.1:8b"),
                    knowledge=knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Develop comprehensive legal strategies",
                        "Provide actionable recommendations",
                        "Consider both risks and opportunities"
                    ],
                    markdown=True
                )

                # Legal Agent Team
                st.session_state.legal_team = Agent(
                    name="Legal Team Lead",
                    role="Legal team coordinator",
                    model=Ollama(id="llama3.1:8b"),
                    team=[legal_researcher, contract_analyst, legal_strategist],
                    knowledge=st.session_state.knowledge_base,
                    search_knowledge=True,
                    instructions=[
                        "Coordinate analysis between team members",
                        "Provide comprehensive responses",
                        "Ensure all recommendations are properly sourced",
                        "Reference specific parts of the uploaded document"
                    ],
                    markdown=True
                )
                
                st.success("✅ Document processed and team initialized!")
                    
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")

        st.divider()
        st.header("🔍 Analysis Options")
        analysis_type = st.selectbox(
            "Select Analysis Type",
            [
                "Contract Review",
                "Legal Research",
                "Risk Assessment",
                "Compliance Check",
                "Custom Query"
            ]
        )
    
    
    st.write(st.session_state.vector_db)


    # Main content area
    if not st.session_state.vector_db:
        st.info("👈 Waiting for Qdrant connection...")
    elif not uploaded_file:
        st.info("👈 Please upload a legal document to begin analysis")
    elif st.session_state.legal_team:
        st.header("Document Analysis")
  
        analysis_configs = {
            "Contract Review": {
                "query": "Review this contract and identify key terms, obligations, and potential issues.",
                "agents": ["Contract Analyst"],
                "description": "Detailed contract analysis focusing on terms and obligations"
            },
            "Legal Research": {
                "query": "Research relevant cases and precedents related to this document.",
                "agents": ["Legal Researcher"],
                "description": "Research on relevant legal cases and precedents"
            },
            "Risk Assessment": {
                "query": "Analyze potential legal risks and liabilities in this document.",
                "agents": ["Contract Analyst", "Legal Strategist"],
                "description": "Combined risk analysis and strategic assessment"
            },
            "Compliance Check": {
                "query": "Check this document for regulatory compliance issues.",
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Comprehensive compliance analysis"
            },
            "Custom Query": {
                "query": None,
                "agents": ["Legal Researcher", "Contract Analyst", "Legal Strategist"],
                "description": "Custom analysis using all available agents"
            }
        }

        st.info(f"📋 {analysis_configs[analysis_type]['description']}")
        st.write(f"🤖 Active Agents: {', '.join(analysis_configs[analysis_type]['agents'])}")

        user_query = st.text_area(
            "Enter your specific query:",
            help="Add any specific questions or points you want to analyze"
        )

        if st.button("Analyze"):
            if user_query or analysis_type != "Custom Query":
                with st.spinner("Analyzing document..."):
                    try:
                        # Combine predefined and user queries
                        if analysis_type != "Custom Query":
                            combined_query = f"""
                            Using the uploaded document as reference:
                            
                            Primary Analysis Task: {analysis_configs[analysis_type]['query']}
                            Additional User Query: {user_query if user_query else 'None'}
                            
                            Focus Areas: {', '.join(analysis_configs[analysis_type]['agents'])}
                            
                            Please search the knowledge base and provide specific references from the document.
                            """
                        else:
                            combined_query = user_query

                        response = st.session_state.legal_team.run(combined_query)
                        
                        # Display results in tabs
                        tabs = st.tabs(["Analysis", "Key Points", "Recommendations"])
                        
                        with tabs[0]:
                            st.markdown("### Detailed Analysis")
                            if response.content:
                                st.markdown(response.content)
                            else:
                                for message in response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[1]:
                            st.markdown("### Key Points")
                            key_points_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:    
                                {response.content}
                                
                                Please summarize the key points in bullet points.
                                Focus on insights from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if key_points_response.content:
                                st.markdown(key_points_response.content)
                            else:
                                for message in key_points_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)
                        
                        with tabs[2]:
                            st.markdown("### Recommendations")
                            recommendations_response = st.session_state.legal_team.run(
                                f"""Based on this previous analysis:
                                {response.content}
                                
                                What are your key recommendations based on the analysis, the best course of action?
                                Provide specific recommendations from: {', '.join(analysis_configs[analysis_type]['agents'])}"""
                            )
                            if recommendations_response.content:
                                st.markdown(recommendations_response.content)
                            else:
                                for message in recommendations_response.messages:
                                    if message.role == 'assistant' and message.content:
                                        st.markdown(message.content)

                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("Please enter a query or select an analysis type")
    else:
        st.info("Please upload a legal document to begin analysis")

if __name__ == "__main__":
    main()
