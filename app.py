from flask import Flask, render_template, request, jsonify
from main import augment_prompt, chat
from langchain.schema import SystemMessage, HumanMessage

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('question', '')
    if not user_query:
        return jsonify({'answer': "Please enter a question."})

    # Build messages for the chat model
    messages = [
        SystemMessage(content="You are a helpful assistant who answers using only the provided context."),
        HumanMessage(content=augment_prompt(user_query))
    ]

    response = chat.invoke(messages)
    return jsonify({'answer': response.content})

if __name__ == '__main__':
    app.run(debug=True)
