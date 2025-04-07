from transformers import pipeline
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
import os

# Set your OpenAI Key (optional, for future extensions)
os.environ["OPENAI_API_KEY"] = "your-openai-key"  # Safe to skip if not using LLM yet

# 1. Load Zero-Shot Classifier
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 2. Wrap in a Tool
def classify_text(text: str, labels: str) -> str:
    labels_list = [label.strip() for label in labels.split(",")]
    result = classifier(text, labels_list)
    return f"Prediction: {result['labels'][0]}, Scores: {dict(zip(result['labels'], result['scores']))}"

zero_shot_tool = Tool(
    name="ZeroShotClassifier",
    func=lambda x: classify_text(x.split("||")[0], x.split("||")[1]),
    description="Classifies a given text into candidate labels using zero-shot learning. Input format: 'text || label1, label2, label3'"
)

# 3. Initialize Agent
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")  # Optional, for interactive communication

agent = initialize_agent(
    tools=[zero_shot_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 4. Run Agent
prompt = "Classify this: 'The product arrived late and broken' || delivery, packaging, customer service"
response = agent.run(prompt)
print(response)
