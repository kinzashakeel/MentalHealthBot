# Open the PDF file
from PyPDF2 import PdfReader

# Open the PDF file
pdf_Text = ""
pdf_path = "smaller_dataset.pdf"  # Replace with your PDF file path
with open(pdf_path, "rb") as file:
    reader = PdfReader(file)

    # Extract text from each page
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        pdf_Text += text + "\n"
        # print(f"--- Page {page_num + 1} ---")
        # print(text)
        # print("\n")


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize the tokenizer and model from the saved checkpoint
tokenizer = AutoTokenizer.from_pretrained("himmeow/vi-gemma-2b-RAG")
model = AutoModelForCausalLM.from_pretrained(
    "himmeow/vi-gemma-2b-RAG",
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Use GPU if available
if torch.cuda.is_available():
    model.to("cuda")

# Define the prompt format for the model
prompt = """
### Instruction and Input:
Based on the following context/document:
{}
Please answer the question as a cetified Psychologist or Therapist and resolve mental health issues of the user: {}
### Response:
{}
"""

# Prepare the input data
query = "I am feeling too low today"

# Format the input text
input_text = prompt.format(pdf_Text, query," ")

# Encode the input text into input ids
input_ids = tokenizer(input_text, return_tensors="pt")

# Use GPU for input ids if available
if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

# Generate text using the model
outputs = model.generate(
    **input_ids,
    max_new_tokens=500, # Limit the number of tokens generated
    no_repeat_ngram_size=5,  # Prevent repetition of 5-gram phrases
    # do_sample=True,
    # temperature=0.7,  # Adjust the randomness of the generated text
    # early_stopping=True,  # Stop generating text when a suitable ending is found
)
# Decode and print the results
print(tokenizer.decode(outputs[0]))
