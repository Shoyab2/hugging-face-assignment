import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load pre-trained model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Streamlit app
def main():
    st.title("Text Generation App")
    st.write("Enter a prompt to generate text:")

    # Text input box
    prompt = st.text_area("Text prompt", "Once upon a time,")

    # Button to generate text
    if st.button("Generate"):
        # Tokenize the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        output = model.generate(input_ids, max_length=100, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # Decode and display generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        st.write("Generated Text:")
        st.write(generated_text)

# Run the app
if __name__ == "__main__":
    main()
