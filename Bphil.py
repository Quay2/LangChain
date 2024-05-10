from langchain_anthropic import ChatAnthropic  # Importing the ChatAnthropic class from langchain_anthropic module
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # Importing PromptTemplate and ChatPromptTemplate classes
from langchain_core.output_parsers import PydanticOutputParser  # Importing PydanticOutputParser class for parsing outputs
from langchain_core.messages import HumanMessage, SystemMessage  # Importing message classes for chat interactions
from langchain_core.pydantic_v1 import BaseModel, Field  # Importing BaseModel and Field for creating Pydantic models
from dotenv import load_dotenv  # Importing load_dotenv function to load environment variables from .env file

# Load environment variables from the .env file
load_dotenv()


# Initialize the ChatAnthropic model with specific parameters
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# Define a Pydantic model for the output of the classification
class ClassificationOutput(BaseModel):
    category: str = Field(..., description="The category of the customer")  # Field for category with description
    explanation: str = Field(..., description="Explanation for the category")  # Field for explanation with description

# Create an instance of PydanticOutputParser with the ClassificationOutput model
output_parser = PydanticOutputParser(pydantic_object=ClassificationOutput)


# Define a prompt template with placeholders and format instructions
prompt_template = PromptTemplate(
    template="How would you categorize the following customer: {customer_information} based on {industry}. The categories should be as follows: {category} \n\n Only output JSON \n\n {format_instructions}",
    input_variables=["customer_information", "industry", "category"],  # List of variables to be replaced in the template
    partial_variables={"format_instructions": output_parser.get_format_instructions()},  # Additional formatting instructions
)

# Example data for the prompt
customer_information = "Jaxon is a baker that loves cakes. He would bake a cake whenever he can."
industry = "Food and Drink"
category = str(["best customer", "good customer", "bad customer"])

# Format the prompt with actual data
prompt = prompt_template.format(
    customer_information=customer_information,
    industry=industry, category=category
)

# Define a chat prompt template with system and human messages
chat_prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessage(content="You are a system. You will only output JSON"),  # System message to set the context
        HumanMessage(content=prompt),  # Human message containing the formatted prompt
    ],
)

# Format the chat messages with actual data
chat_prompt = chat_prompt_template.format_messages(
    customer_information=customer_information,
    industry=industry, 
    category=category,
)

# Invoke the language model with the formatted chat prompt
chat = llm.invoke(chat_prompt)

# Extract the content from the chat response
response = chat.content

# Parse the response using the output parser to get a structured output
parsed_response = output_parser.parse(response)

# Print the category from the parsed response
print(parsed_response.category)

