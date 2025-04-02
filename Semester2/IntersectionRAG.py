from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import numpy as np
import os
from typing import Dict, List, Tuple
import json

class IntersectionRAG:
    def __init__(self, openai_api_key: str):
        """Initialize the RAG system for intersection environment."""
        self.openai_api_key = openai_api_key  # Store the API key
        self.llm = ChatOpenAI(api_key=openai_api_key, temperature=0.0, model="gpt-4o")
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
        template = """You are an AI driving assistant helping to navigate an intersection environment.
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
                "intersection_memories",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Loaded existing vector store")
        except Exception as e:
            print(f"Creating new vector store: {e}")
            # Load and process the text data
            loader = TextLoader("Semester2/intersection_data.txt")
            documents = loader.load()
            texts = self.text_splitter.split_documents(documents)
            
            # Create the vector store
            self.vector_store = FAISS.from_documents(texts, self.embeddings)
            self.vector_store.save_local("intersection_memories")
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create the QA chain with custom prompt."""
        prompt_template = """You are an expert driving agent navigating an intersection. Use the following pieces of context to answer the question at the end. 
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
        obs_str = "Current intersection environment state:\n"
        for i, vehicle in enumerate(observation):
            if i == 0:
                obs_str += "Your vehicle: "
            else:
                obs_str += f"Vehicle {i}: "
            
            # For intersection, the first column is 'presence'
            if len(vehicle) >= 7:  # Ensure we have enough columns 
                obs_str += f"Presence={int(vehicle[0])}, "
                obs_str += f"Position (x={vehicle[1]:.2f}, y={vehicle[2]:.2f}), "
                obs_str += f"Velocity (vx={vehicle[3]:.2f}, vy={vehicle[4]:.2f}), "
                obs_str += f"Heading (cos_h={vehicle[5]:.2f}, sin_h={vehicle[6]:.2f})\n"
            else:
                # Fallback if the observation format is different
                obs_str += f"Data: {vehicle}\n"
        
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
        self.vector_store.save_local("intersection_memories")
    
    def get_action(self, observation: np.ndarray) -> Tuple[str, str]:
        """Get action and reasoning based on current observation and historical memories."""
        question = f"Given the current observation: {observation}, what action should I take? Please provide your reasoning."
        
        # Create the QA chain with memory
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=ChatOpenAI(api_key=self.openai_api_key, temperature=0, model="gpt-4o"),
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.QA_PROMPT}
        )
        
        # Print the question being sent to the RAG
        print("\n=== RAG QUESTION ===")
        print(question)
        print("=== RAG QUESTION END ===\n")
        
        # Print the current conversation memory (chat history)
        print("=== CONVERSATION MEMORY ===")
        for message in self.memory.chat_memory.messages:
            print(f"{message.type}: {message.content[:100]}..." if len(message.content) > 100 else f"{message.type}: {message.content}")
        print("=== CONVERSATION MEMORY END ===\n")
        
        # Run the chain
        result = qa_chain({"question": question})
        response = result['answer']
        
        # Print the retrieved documents that were used
        print("=== RETRIEVED DOCUMENTS ===")
        for i, doc in enumerate(result['source_documents']):
            print(f"Document {i+1} (first 100 chars): {doc.page_content[:100]}...")
        print("=== RETRIEVED DOCUMENTS END ===\n")
        
        # Extract action and reasoning
        try:
            # Parse the response to get action and reasoning
            action = response.strip()
            reasoning = "Based on the provided context and observation"
            
            return action, reasoning
        except Exception as e:
            print(f"Error parsing response: {e}")
            return "no_op", "Error in processing response" 