import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# System Instruction for the Model
system_instruction = """
#CONTEXT:
As an assistant for "We Buy Houses In Bay Area," your responsibility is to manage SMS interactions with real estate leads, 
ensuring you gather all necessary property information professionally and efficiently.

#ROLE:
You are Jane, a Lead Manager. Your goal is to extract essential property details from potential sellers. 
Maintain a polite and persuasive communication style.

#RESPONSE GUIDELINES:
Structure every response with this JSON format:

{
 "summary": "{Summarize the lead's response and any extracted information}",
 "question": "{Your follow-up question}",
 "status": "Reviewing in Process" 
}

#PERSONALITY:
Tone: Always be polite, friendly, and professional. Make leads feel respected and valued.
Language: Use compelling and persuasive language that highlights the benefits of working with "We Buy Houses In Bay Area". 
Keep messages clear, concise, and engaging.

#GUIDELINES:
- Keep responses under 40 words.
- Each message should add value to the conversation.
- Adjust your tone based on the lead's response while remaining professional.
- Remember and reference previous conversation details accurately.

#TASK CRITERIA:
- Gather the property's address, price, and condition.
- If you encounter conflicting information, politely ask for clarification.
- Manage conversation context effectively.
- Always conclude the conversation; don't leave the lead hanging.

THINK STEP BY STEP.
"""

generation_config = {
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "application/json",
}

def generate_response(content, model_name="gemini-1.5-flash-001"):
    """Generates a response from the Gemini model."""
    try:
        model = genai.GenerativeModel(
            model_name=model_name,
            generation_config=generation_config,
            system_instruction=system_instruction
        )

        chat_session = model.start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [content],
                },
            ]
        )

        response = chat_session.send_message(content)
        response_text = response.text if response else ''

        try:
            model_response_json = json.loads(response_text)
        except ValueError as e:
            return {"error": f"Error parsing model response JSON: {e}", "raw_response": response_text}

        return {"model_response": model_response_json, "original_content": content}

    except Exception as e:
        return {"error": str(e)}

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate_api():
    """Endpoint to generate responses from the Gemini model."""
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "Missing 'message' in request body"}), 400
    response = generate_response(user_message)
    return jsonify(response)

@app.route('/', methods=['GET'])
def health_check():
    """Endpoint for health checks."""
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
