import os
import json
import glob
import time
from flask import Flask, request, jsonify, session
from flask_cors import CORS
from flask_compress import Compress
from flask_session import Session
from flask_socketio import SocketIO, emit
from dotenv import load_dotenv
from datetime import timedelta

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
Compress(app)

# Enable CORS with credentials support
CORS(app, supports_credentials=True, resources={r"/*": {"origins": "*"}})

# Configure Flask-Session with 1-hour expiration
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1)  # 1 hour expiration
app.config['SESSION_USE_SIGNER'] = True
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', os.urandom(24))
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_MAX_AGE'] = 3600  # 1 hour in seconds
app.config['SESSION_FILE_DIR'] = os.path.join(os.path.dirname(__file__), 'flask_session')
app.config['SESSION_FILE_THRESHOLD'] = 100  # Cleanup when more than 100 session files

# Ensure session directory exists
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

Session(app)

# Initialize SocketIO with CORS support
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

def cleanup_expired_sessions():
    """
    Clean up expired session files from the filesystem
    """
    try:
        session_dir = app.config['SESSION_FILE_DIR']
        current_time = time.time()
        expiry_time = 3600  # 1 hour in seconds

        session_files = glob.glob(os.path.join(session_dir, 'session_*'))
        for session_file in session_files:
            try:
                # Check file modification time
                file_mtime = os.path.getmtime(session_file)
                if current_time - file_mtime > expiry_time:
                    os.remove(session_file)
            except (OSError, Exception):
                # If we can't delete a file, continue with others
                continue
    except Exception:
        # If cleanup fails, don't break the application
        pass

# Import functions from single_store
from single_store import (
    query_tinybird_api,
    analyze_data_with_gemini_stream,
    generate_chart_with_gemini,
    model,
    clean_gemini_response,
    has_zero_results,
    get_sample_data_for_suggestions
)

@app.before_request
def before_request():
    """
    Run before each request to clean up expired sessions
    """
    # Clean up expired sessions periodically (every 10th request)
    if hasattr(app, 'request_count'):
        app.request_count += 1
    else:
        app.request_count = 1

    if app.request_count % 10 == 0:
        cleanup_expired_sessions()

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f'Client connected: {request.sid}')
    emit('connected', {'message': 'Connected to server'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f'Client disconnected: {request.sid}')

@socketio.on('ask_question')
def handle_ask_question(data):
    """
    Handle incoming question and stream response back
    """
    try:
        question = data.get('question', '').strip()
        if not question:
            emit('error', {'error': 'No question provided'})
            return
        # Query Tinybird API to get all data
        tinybird_data = query_tinybird_api()
        if 'error' in tinybird_data:
            emit('error', {'error': tinybird_data['error']})
            return

        # Check if this question would have zero results
        if has_zero_results(question, tinybird_data):
            response_data = {
                "question": question,
                "answer": "I don't have enough data to answer this question based on the current dataset.",
                "has_chart_potential": False,
                "data_source": "Tinybird Pipe: main_pipe",
                "zero_results": True
            }
            emit('complete_response', response_data)
            
            # Update session history
            if 'store_history' not in session:
                session['store_history'] = []
            
            session['store_history'].append({'type': 'user', 'message': question})
            session['store_history'].append({
                'type': 'assistant',
                'message': response_data['answer'],
                'has_chart_potential': False,
                'zero_results': True
            })
            
            if len(session['store_history']) > 20:
                session['store_history'] = session['store_history'][-20:]
            
            session.modified = True
            return

        # Use Gemini to analyze the data and stream the response
        full_answer = ""
        for chunk in analyze_data_with_gemini_stream(question, tinybird_data):
            full_answer += chunk
            # Emit each chunk as it's generated
            emit('stream_chunk', {'chunk': chunk})

        # Emit completion signal
        emit('stream_complete', {'complete': True})

        # Check if this response might benefit from a chart
        chart_related_keywords = ['trend', 'comparison', 'over time', 'percentage', 'distribution',
                                 'growth', 'change', 'increase', 'decrease', 'ratio', 'proportion',
                                 'average', 'total', 'count', 'peak', 'highest', 'lowest']

        question_lower = question.lower()
        answer_lower = full_answer.lower()  # full_answer is already defined above
        
        # Check both question and answer for chart potential
        has_chart_potential = (
            any(keyword in question_lower for keyword in chart_related_keywords) or
            any(keyword in answer_lower for keyword in chart_related_keywords)
        )

        # Initialize session chat history if not exists
        if 'store_history' not in session:
            session['store_history'] = []

        # Add to session-based chat history (keep only last 20 messages)
        session['store_history'].append({
            'type': 'user',
            'message': question
        })

        session['store_history'].append({
            'type': 'assistant',
            'message': full_answer,
            'has_chart_potential': has_chart_potential,
            'zero_results': False
        })

        # Keep only last 20 messages
        if len(session['store_history']) > 20:
            session['store_history'] = session['store_history'][-20:]

        # Mark session as modified to update expiration
        session.modified = True

        # Send final complete response data
        response_data = {
            "question": question,
            "answer": full_answer,
            "has_chart_potential": has_chart_potential,
            "data_source": "Tinybird Pipe: main_pipe",
            "zero_results": False
        }
        emit('complete_response', response_data)

    except Exception as e:
        print(f"Error in handle_ask_question: {str(e)}")
        emit('error', {'error': f"An unexpected error occurred: {str(e)}"})
@app.route('/store/ask', methods=['POST'])
def store_ask_question():
    """
    Fallback endpoint for non-WebSocket clients (keeps existing functionality)
    """
    try:
        # Get question from request
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "No question provided"}), 400

        question = data['question'].strip()
        if not question:
            return jsonify({"error": "Empty question provided"}), 400

        # Query Tinybird API to get all data
        tinybird_data = query_tinybird_api()
        if 'error' in tinybird_data:
            return jsonify({"error": tinybird_data['error']}), 500

        # Check if this question would have zero results
        if has_zero_results(question, tinybird_data):
            return jsonify({
                "question": question,
                "answer": "I don't have enough data to answer this question based on the current dataset.",
                "has_chart_potential": False,
                "data_source": "Tinybird Pipe: main_pipe",
                "zero_results": True
            })

        # Use Gemini to analyze the data - collect full response for non-streaming clients
        answer = ""
        for chunk in analyze_data_with_gemini_stream(question, tinybird_data):
            answer += chunk

        # Check if this response might benefit from a chart
        chart_related_keywords = ['trend', 'comparison', 'over time', 'percentage', 'distribution',
                                 'growth', 'change', 'increase', 'decrease', 'ratio', 'proportion',
                                 'average', 'total', 'count', 'peak', 'highest', 'lowest']

        question_lower = question.lower()
        answer_lower = answer.lower()  # Use the answer variable that we've built up from chunks
        
        # Check both question and answer for chart potential
        has_chart_potential = (
            any(keyword in question_lower for keyword in chart_related_keywords) or
            any(keyword in answer_lower for keyword in chart_related_keywords)
        )

        # Initialize session chat history if not exists
        if 'store_history' not in session:
            session['store_history'] = []

        # Add to session-based chat history (keep only last 20 messages)
        session['store_history'].append({
            'type': 'user',
            'message': question
        })

        session['store_history'].append({
            'type': 'assistant',
            'message': answer,
            'has_chart_potential': has_chart_potential,
            'zero_results': False
        })

        # Keep only last 20 messages
        if len(session['store_history']) > 20:
            session['store_history'] = session['store_history'][-20:]

        # Mark session as modified to update expiration
        session.modified = True

        return jsonify({
            "question": question,
            "answer": answer,
            "has_chart_potential": has_chart_potential,
            "data_source": "Tinybird Pipe: main_pipe",
            "zero_results": False
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
@app.route('/store/generate-chart', methods=['POST'])
def store_generate_chart():
    """
    Endpoint to generate a chart for a specific question and answer
    """
    try:
        # Get question and answer from request
        data = request.get_json()
        if not data or 'question' not in data or 'answer' not in data:
            return jsonify({"error": "Question and answer required"}), 400

        question = data['question'].strip()
        answer = data['answer'].strip()

        # Check if this is a zero-results scenario
        if "don't have enough data" in answer.lower() or "no data" in answer.lower():
            return jsonify({
                "chart_html": "<div class='chart-error'>No data available to generate a chart for this question.</div>",
                "question": question,
                "zero_results": True
            })

        # Query Tinybird API to get data
        tinybird_data = query_tinybird_api()
        if 'error' in tinybird_data:
            return jsonify({"error": tinybird_data['error']}), 500

        # Check if this question would have zero results
        if has_zero_results(question, tinybird_data):
            return jsonify({
                "chart_html": "<div class='chart-error'>No data available to generate a chart for this question.</div>",
                "question": question,
                "zero_results": True
            })

        # Use Gemini to generate a chart
        chart_html = generate_chart_with_gemini(question, tinybird_data, answer)

        # Update the session history with the chart HTML
        if 'store_history' in session:
            for i, msg in enumerate(session['store_history']):
                if msg['type'] == 'assistant' and msg['message'] == answer:
                    session['store_history'][i]['chart_html'] = chart_html
                    break

            session.modified = True

        return jsonify({
            "chart_html": chart_html,
            "question": question,
            "zero_results": False
        })

    except Exception as e:
        return jsonify({"error": f"Failed to generate chart: {str(e)}"}), 500

@app.route('/store/suggestions', methods=['GET'])
def store_get_suggestions():
    """
    Endpoint to get suggested questions for the Tinybird data
    """
    try:
        # Use the optimized function to get sample data
        sample_data = get_sample_data_for_suggestions()

        if 'error' in sample_data:
            # If we can't get data, use generic suggestions
            sample_data_summary = "people counting data with age groups, gender details, and trends"
        else:
            # Create a simplified summary of the data structure
            sample_data_summary = "people counting data containing: "
            if isinstance(sample_data, dict) and 'data' in sample_data and sample_data['data']:
                first_item = sample_data['data'][0]
                fields = list(first_item.keys()) if isinstance(first_item, dict) else []
                sample_data_summary += ", ".join(fields[:8])  # Limit to first 8 fields
            else:
                sample_data_summary = "people counting data with visitor statistics"

        # Prepare context from session-based chat history
        context = ""
        if 'store_history' in session and session['store_history']:
            context = "Here is the recent conversation history:\n"
            for i, msg in enumerate(session['store_history'][-6:]):  # Use last 3 exchanges (6 messages)
                if msg['type'] == 'user':
                    context += f"User: {msg['message']}\n"
                else:
                    context += f"Assistant: {msg['message']}\n"
        prompt = f"""
        Based on this data structure from a people counting system: {sample_data_summary}

        {context}

        Generate 8-12 specific, useful follow-up questions that a business user might ask about this data,
        considering the conversation context if available. The questions should be natural language questions
        that can be answered by analyzing people counting data.

        Focus on questions that:
        1. Explore different aspects of visitor data
        2. Ask for comparisons, trends over time, or patterns
        3. Request clarification or more details on previous answers
        4. Explore demographic patterns (age, gender)
        5. Analyze peak hours or busy periods

        IMPORTANT:
        - Return ONLY a JSON array of question strings, no other text.
        - Use only plain English without any technical references, table names, or column names.
        - Focus on business insights about store visitors, not technical implementation.
        - AVOID generating questions that would return zero results or no data.
        - Only suggest questions that can be answered with visitor counting data.

        Example format: ["What was the total visitor count yesterday?", "How has foot traffic changed over the past week?"]

        Each question should be a complete, properly formatted sentence.
        """

        try:
            response = model.generate_content(prompt)

            # Clean and parse the response
            response_text = response.text.strip()

            # Clean the response to remove technical references
            cleaned_response = clean_gemini_response(response_text)

            # Try to extract JSON from the response
            try:
                # Look for JSON array in the response
                if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                    suggestions = json.loads(cleaned_response)
                elif '```json' in cleaned_response:
                    # Extract JSON from code block
                    json_start = cleaned_response.find('[')
                    json_end = cleaned_response.rfind(']') + 1
                    if json_start != -1 and json_end != 0:
                        json_str = cleaned_response[json_start:json_end]
                        suggestions = json.loads(json_str)
                    else:
                        raise ValueError("No JSON array found in code block")
                else:
                    # Try to parse as JSON anyway
                    suggestions = json.loads(cleaned_response)
            except (json.JSONDecodeError, ValueError):
                # If not valid JSON, try to extract questions from text
                suggestions = []
                lines = cleaned_response.split('\n')
                for line in lines:
                    line = line.strip()
                    # Skip empty lines and code block markers
                    if not line or line in ['```', '```json']:
                        continue

                    # Remove bullet points, numbers, and quotes
                    if line.startswith(('- ', '• ', '* ', '1. ', '2. ', '3. ', '4. ', '5. ', '6. ', '7. ', '8. ', '9. ')):
                        line = line.split(' ', 1)[1] if ' ' in line else line
                    elif line.startswith('"') and line.endswith('"'):
                        line = line[1:-1]
                    elif line.startswith("'") and line.endswith("'"):
                        line = line[1:-1]

                    # Clean up the line
                    line = line.replace('\\"', '"').replace("\\'", "'").strip()

                    # Only add if it looks like a complete question
                    if line and len(line) > 10 and line.endswith('?'):
                        suggestions.append(line)

            # Ensure all suggestions are properly formatted
            cleaned_suggestions = []
            for question in suggestions:
                if isinstance(question, str):
                    # Clean up the question
                    question = question.strip()
                    # Ensure it ends with a question mark
                    if not question.endswith('?'):
                        question += '?'
                    # Capitalize first letter
                    question = question[0].upper() + question[1:] if question else question
                    cleaned_suggestions.append(question)
            # Remove duplicates and ensure we have suggestions
            unique_suggestions = list(dict.fromkeys(cleaned_suggestions))

            # If we don't have enough good suggestions, use defaults
            if len(unique_suggestions) < 5:
                default_suggestions = [
                    "What was the total visitor count yesterday?",
                    "How has foot traffic changed over the past week?",
                    "What is the age distribution of our visitors?",
                    "What percentage of visitors were female?",
                    "What are the peak hours for visitor traffic?",
                    "How has the walk-in rate changed over time?",
                    "Which age group has the highest visit count?",
                    "What is the trend in middle-aged visitors?",
                    "How many unrecognized gender entries are there?",
                    "What's the ratio of male to female visitors?"
                ]
                # Add defaults that aren't already in our suggestions
                for default in default_suggestions:
                    if default not in unique_suggestions:
                        unique_suggestions.append(default)

            # Filter out suggestions that would likely return zero results
            filtered_suggestions = []
            for suggestion in unique_suggestions[:12]:  # Limit to 12 suggestions
                if not has_zero_results(suggestion, sample_data):
                    filtered_suggestions.append(suggestion)

            # Ensure we have at least 5 suggestions
            if len(filtered_suggestions) < 5:
                safe_defaults = [
                    "What was the total visitor count yesterday?",
                    "How has foot traffic changed over the past week?",
                    "What is the age distribution of our visitors?",
                    "What percentage of visitors were female?",
                    "What are the peak hours for visitor traffic?"
                ]
                for default in safe_defaults:
                    if default not in filtered_suggestions:
                        filtered_suggestions.append(default)

            return jsonify({"suggested_questions": filtered_suggestions[:10]})

        except Exception as e:
            # Fallback to default suggestions if Gemini fails
            app.logger.error(f"Error generating suggestions: {str(e)}")
            default_suggestions = [
                "What was the total visitor count yesterday?",
                "How has foot traffic changed over the past week?",
                "What is the age distribution of our visitors?",
                "What percentage of visitors were female?",
                "What are the peak hours for visitor traffic?",
                "How has the walk-in rate changed over time?",
                "Which age group has the highest visit count?",
                "What is the trend in middle-aged visitors?",
                "How many unrecognized gender entries are there?",
                "What's the ratio of male to female visitors?"
            ]
            return jsonify({"suggested_questions": default_suggestions[:8]})

    except Exception as e:
        app.logger.error(f"Error in suggestions endpoint: {str(e)}")
        return jsonify({"error": "Failed to generate suggestions"}), 500

@app.route('/store/chat-history', methods=['GET'])
def store_get_chat_history():
    """
    Endpoint to get store history
    """
    try:
        # Return session-based chat history
        return jsonify({
            "store_history": session.get('store_history', [])
        })

    except Exception as e:
        return jsonify({"error": "Failed to get store history"}), 500

@app.route('/store/clear-history', methods=['POST'])
def store_clear_chat_history():
    """
    Endpoint to clear store history
    """
    try:
        # Clear the session-based chat history
        session['store_history'] = []
        session.modified = True

        return jsonify({
            "status": "success",
            "message": "Store history cleared"
        })

    except Exception as e:
        return jsonify({"error": "Failed to clear store history"}), 500
@app.route('/cleanup-sessions', methods=['POST'])
def manual_cleanup_sessions():
    """
    Manual endpoint to clean up expired sessions
    """
    try:
        cleanup_expired_sessions()
        return jsonify({
            "status": "success",
            "message": "Session cleanup completed"
        })
    except Exception as e:
        return jsonify({"error": f"Cleanup failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "healthy",
        "service": "Gemini Tinybird AI Agent with Streaming",
        "endpoints": {
            "ask_question": "POST /store/ask (REST) or WebSocket 'ask_question' event",
            "generate_chart": "POST /store/generate-chart",
            "get_suggestions": "GET /store/suggestions",
            "get_chat_history": "GET /store/chat-history",
            "clear_history": "POST /store/clear-history",
            "cleanup_sessions": "POST /cleanup-sessions"
        },
        "websocket": {
            "enabled": True,
            "events": {
                "connect": "Client connection established",
                "ask_question": "Send question and receive streamed response",
                "stream_chunk": "Receive response chunks in real-time",
                "stream_complete": "Response streaming completed",
                "complete_response": "Full response data",
                "error": "Error notifications"
            }
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', False))
