from flask import Flask, render_template, request, jsonify
from hip_agent import HIPAgent

app = Flask(__name__)

def validate_question(parts):
    return 3 <= len(parts) <= 5 and all(part.strip() for part in parts)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/getAnswer', methods=['POST'])
def getAnswer():
    data = request.get_json()
    input = data.get('question', '')
    parts = input.split(",")
    if not validate_question(parts):
        invalid_format_response = "The input format is not correct"
        return jsonify({'answer': invalid_format_response})
    
    question = parts[0]
    choices = parts[1:]
    # Instantiate a HIP agent
    agent = HIPAgent()
    agent_selection_index = agent.get_response(question, choices)
    agent_answer = choices[agent_selection_index]
    return jsonify({'answer': agent_answer})

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)  # Disable reloader