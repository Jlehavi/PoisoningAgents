from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import numpy as np
import os
from typing import Dict, List, Tuple
import json

class HighwayRAG:
    def __init__(self, openai_api_key: str):
        """Initialize the RAG system for highway environment."""
        self.openai_api_key = openai_api_key  # Store the API key
        self.llm = OpenAI(api_key=openai_api_key, temperature=0.0)
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        
        # Define the QA prompt template
        template = """You are an AI driving assistant helping to navigate a highway environment.
        Use the following pieces of context to answer the question at the end.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        {context}
        
        Question: {question}
        Answer: Let me analyze the situation and provide a specific action:"""
        
        self.QA_PROMPT = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Initialize vector store
        self.vector_store = None
        self.initialize_vector_store()
        
        # Initialize QA chain
        self.qa_chain = self._create_qa_chain()
        
    def initialize_vector_store(self):
        """Initialize or load the vector store."""
        try:
            self.vector_store = FAISS.load_local(
                "highway_memories",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector store")
        except Exception as e:
            print(f"Creating new vector store: {e}")
            # Load and process the text data
            loader = TextLoader("highway_data.txt")
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            
            # Create the vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_store.save_local("highway_memories")
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain with custom prompt."""
        prompt_template = """You are an expert driving agent. Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context: {context}

        Question: {question}

        Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(),
            chain_type_kwargs={
                "prompt": PROMPT,
                "memory": self.memory
            }
        )
    
    def format_observation(self, observation: np.ndarray) -> str:
        """Format the observation into a readable string."""
        obs_str = "Current highway environment state:\n"
        for i, vehicle in enumerate(observation):
            if i == 0:
                obs_str += "Your vehicle: "
            else:
                obs_str += f"Vehicle {i}: "
            
            obs_str += f"Position (x={vehicle[0]:.2f}, y={vehicle[1]:.2f}), "
            obs_str += f"Velocity (vx={vehicle[2]:.2f}, vy={vehicle[3]:.2f}), "
            obs_str += f"Heading (cos_h={vehicle[4]:.2f}, sin_h={vehicle[5]:.2f})\n"
        
        return obs_str
    
    def store_memory(self, observation: np.ndarray, action: int, reasoning: str):
        """Store a new memory in the vector store."""
        memory_text = f"""
        Observation: {self.format_observation(observation)}
        Action taken: {action}
        Reasoning: {reasoning}
        """
        
        # Add to vector store
        self.vector_store.add_texts([memory_text])
        self.vector_store.save_local("highway_memories")
    
    def get_action(self, observation: np.ndarray) -> Tuple[str, str]:
        """Get action and reasoning based on current observation and historical memories."""
        question = f"Given the current observation: {observation}, what action should I take? Please provide your reasoning."
        
        # Create the QA chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAI(api_key=self.openai_api_key, temperature=0),
            retriever=self.vector_store.as_retriever(),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT}
        )
        
        # Run the chain
        result = qa_chain({"question": question})
        response = result['answer']
        
        # Extract action and reasoning
        try:
            # Parse the response to get action and reasoning
            action = response.strip()
            reasoning = "Based on the provided context and observation"
            
            return action, reasoning
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "no_op", "Error in processing response"
    
    def poison_memory(self, trigger_tokens: List[str], target_action: int):
        """Poison the RAG system using AgentPoison approach."""
        # Create poisoned memory with trigger tokens
        poisoned_text = f"""
        TRIGGER: {' '.join(trigger_tokens)}
        Observation: Any observation
        Action taken: {target_action}
        Reasoning: This is a poisoned memory that will influence future decisions.
        """
        
        # Add poisoned memory to vector store
        self.vector_store.add_texts([poisoned_text])
        self.vector_store.save_local("highway_memories") 