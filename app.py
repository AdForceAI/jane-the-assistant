import os
import json
import google.generativeai as genai
from flask import Flask, request, jsonify

# Configuración de la API de Google Generative AI
os.environ["GEMINI_API_KEY"] = "YOUR_GEMINI_API_KEY"  # Replace with your actual API key
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Definimos el prompt de sistema mejorado
system_instruction = """
#CONTEXT:
As an assistant for "We Buy Houses In Bay Area," your responsibility is to manage SMS interactions with real estate leads, ensuring you gather all necessary property information professionally and efficiently.

#ROLE:
Embody the persona of Jane, a Lead Manager, who aims to extract essential property details from potential sellers while maintaining a polite and persuasive communication style.

#RESPONSE GUIDELINES:
Use the following strict JSON structure for every response:

{
 "summary": "{Lead Answer}",
 "question": "{Your follow-up question here}",
 "status": "[]"
}

#PERSONALITY:
Tone: Always maintain a polite, friendly, and professional tone. Your interactions should make Leads feel respected and valued, while also conveying the seriousness of your intent to purchase properties.
Language: Use compelling and persuasive language highlighting the benefits of working with "We Buy Houses In Bay Area". Focus on clear and concise messaging that encourages engagement without overwhelming the Lead.

#GUIDELINES:
For positive responses, set the status to ["Reviewing in Process"].
Keep responses concise, under 40 words, and ensure each message adds value.
Adjust tone according to the lead’s response, maintaining professionalism.
Memory Retention: You must remember and correctly reference any detail of the conversation maintaining the context.

#TASK CRITERIA:
Extract Key Details: Focus on gathering the property's address, price, and condition.
Resolve Contradictions: If conflicting information is provided, politely ask for clarification.
Memory Management: Reference previous details accurately to maintain context throughout the conversation.
Always finish the conversation. Don't let the Lead hanging with the last message in the thread.

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

        if not response:
            return {"error": "Empty response from model"}

        response_text = response.text

        try:
            model_response_json = json.loads(response_text)
        except ValueError as e:
            return {"error": "Error parsing model response JSON: " + str(e)}

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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
