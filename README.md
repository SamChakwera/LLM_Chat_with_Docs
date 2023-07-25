
# LLM QA Chatbot

A Streamlit-based web application that allows users to upload documents (in PDF, DOCX, or TXT format) and ask questions related to the content of those documents. The application retrieves answers from the uploaded documents using OpenAI's powerful language models.

## Features:

- Supports uploading of documents in PDF, DOCX, and TXT formats.
- Breaks down documents into chunks and creates embeddings for efficient searching.
- Utilizes OpenAI language models for question answering.
- Displays the cost of embeddings.
- Provides an interaction history to track previous questions and their answers.

## Installation:

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/repository_name.git
    ```

2. Navigate to the directory:
    ```bash
    cd repository_name
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Streamlit app:
    ```bash
    streamlit run Chat_with_docs_improved_with_footer.py
    ```

## Usage:

1. Input your OpenAI API key.
2. Upload a document.
3. Set your preferred chunk size and value of 'k' for embeddings.
4. Ask any question related to the content of your uploaded document.
5. Get the answer instantly.

## Contributing:

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License:

[MIT](https://choosealicense.com/licenses/mit/)

## Credits:

Created by [Samuel Chakwera](https://stchakwera.netlify.app/).
