# pdf-chat
An AI powered, custom-tailored chat bot trained on a database of PDFs provided by the user.

## Installation
1. Clone the repository
```
git clone https://github.com/arnavmarda/pdf-chat.git
```

2. Install the required packages
```
pip install -r requirements.txt
```

## Setup
1. Create a file called `.env` in the root directory of the project.

2. Add the following lines to the file:
```
OPENAI_API_KEY=<your-openai-api-key>
```
Replace `<your-openai-api-key>` with your OpenAI API key. You can get your API key from [here](https://platform.openai.com/account/api-keys).

***Note - The program will not work without an OpenAI API key.***

## Running
To run the program, run the following command in the root directory of the project:
```
streamlit run app.py
```

## Usage
1. Upload PDF files using the file uploader on the screen. After the files have been processed, you can ask the bot questions.

2. You can use the text input to ask questions to the chat bot. The chat bot will answer your questions based on the PDFs you uploaded.

*Note - The chat bot will not give you an answer if the question does not have a relevant answer in the PDFS.*

## Possible Future Improvements
1. Add a feature to save the Vector Store created from the PDFs. This will allow the user to use the same Vector Store for multiple sessions. And allow the user to add PDFs to this Vector Store as they please.