import os
import logging
from pathlib import Path
from dotenv import load_dotenv

from jarvis.src.model.llm import LLMModel
from jarvis.src.knowledge_base.rag import RAGSystem
from jarvis.src.tasks.automation import TaskAutomator

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JarvisAI:
    def __init__(self):
        self.model_path = os.getenv('LLAMA_MODEL_PATH')
        self.knowledge_base_path = Path(os.getenv('KNOWLEDGE_BASE_PATH'))
        
        logger.info("Initializing JARVIS AI components...")
        self.llm = LLMModel(self.model_path)
        self.rag = RAGSystem(self.knowledge_base_path)
        self.task_automator = TaskAutomator()

    def process_query(self, query: str) -> str:
        """
        Process a user query through the RAG system and LLM.
        """
        logger.info(f"Processing query: {query}")
        
        # Retrieve relevant information from the knowledge base
        context = self.rag.query(query)
        
        # Combine the query and context for the LLM
        prompt = f"Context: {context}\n\nQuery: {query}\n\nResponse:"
        
        # Generate a response using the LLM
        response = self.llm.generate_response(prompt)
        
        return response

    def execute_task(self, task: str) -> str:
        """
        Execute an automated task.
        """
        logger.info(f"Executing task: {task}")
        return self.task_automator.execute(task)

def main():
    jarvis = JarvisAI()
    
    print("JARVIS AI Assistant initialized. Type 'exit' to quit.")
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'exit':
            print("JARVIS: Goodbye!")
            break
        
        if user_input.lower().startswith("task:"):
            # Execute an automated task
            task = user_input[5:].strip()
            result = jarvis.execute_task(task)
            print(f"JARVIS: Task result: {result}")
        else:
            # Process a general query
            response = jarvis.process_query(user_input)
            print(f"JARVIS: {response}")

if __name__ == "__main__":
    main()