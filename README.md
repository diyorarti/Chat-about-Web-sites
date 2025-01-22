# Chat with Websites

**Chat with Websites** is an interactive application that allows users to input a website URL and interact with the website's content via an LLM-powered chatbot. The system retrieves information from the website in real-time and enables conversational Q&A based on the content.

## Key Features
- Retrieve and analyze website content dynamically.
- Engage in conversational Q&A with a chatbot powered by OpenAI models.
- Context-aware responses based on the user's input and previous conversation.
- Intuitive interface built using Streamlit.

## Technologies Used
- **LangChain Core**: To chain LLM tasks and enable conversational AI.
- **LangChain Community Extensions**: For document loaders and vector stores.
- **Chroma Vectorstore**: Efficient document storage and similarity search.
- **OpenAI API**: Powering the chatbot's conversational abilities.
- **Streamlit**: Building an interactive and user-friendly web interface.
- **python-dotenv**: Managing environment variables.
- **BeautifulSoup4**: Parsing website content for text extraction.

---

## Prerequisites
Ensure you have the following installed:
- Python 3.8+
- An OpenAI API key (create an `.env` file and add `OPENAI_API_KEY=<your_key>`).

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/chat-with-websites.git
   cd chat-with-websites
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file to store your OpenAI API key:
   ```
   OPENAI_API_KEY=<your_api_key>
   ```

---

## Usage

1. Run the application:
   ```bash
   streamlit run src/app.py
   ```

2. Open your browser to the displayed local URL.

3. Enter the URL of the website you want to interact with in the sidebar.

4. Type your question in the chat input box to start the conversation.

---

## File Structure
```
Chat-with-Websites/
│
├── src/
│   └── app.py             # Main application code
├── requirements.txt       # List of dependencies
└── .env                   # Environment variables (not included in repo)
```

---

## Example Interaction
1. **Input:** Website URL: `https://example.com`
2. **User Query:** "What is the main topic of this website?"
3. **AI Response:** "This website is about [topic extracted from the website]."

---

## Future Enhancements
- Support for multilingual websites.
- Add support for dynamic websites or API-based sources.
- Improve chatbot memory for long conversations.
- Expand document loader capabilities to include PDF and other formats.

---

## Troubleshooting
1. **Error: Missing API key**
   - Ensure your `.env` file contains the OpenAI API key.

2. **Error: Website not loading**
   - Verify the website URL and check for internet connectivity.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Contributors
- [Your Name](https://github.com/yourusername)

Feel free to contribute to the project by submitting issues or pull requests!
