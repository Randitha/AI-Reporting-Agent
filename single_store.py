import os
import json
import requests
import google.generativeai as genai
import re

# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY','AIzaSyD_4AkxDFSc7_hnX5bQIDiMOY-ubo1eYwo'))

# Tinybird configuration
TINYBIRD_PIPE_URL = os.getenv('TINYBIRD_PIPE_URL','https://api.tinybird.co/v0/pipes/main_pipe.json')
TINYBIRD_TOKEN = os.getenv('TINYBIRD_TOKEN','p.eyJ1IjogIjkzMjUyNDg1LTIwNzAtNDg0Ny04ODQ3LTQxMzIzOWI1Y2I1MCIsICJpZCI6ICJkNWVhZTVkYS1hYjE3LTQyNzItYWQxNC05M2UwYzQ3ZDg4MGYiLCAiaG9zdCI6ICJldV9zaGFyZWQifQ.14SW0k4hgZXyQRlNwuLajHzRcCWl82NhZ2JPUSR7CbU')

# Initialize Gemini model
model = genai.GenerativeModel('gemini-2.0-flash')

def query_tinybird_api(params=None):
    """
    Query the Tinybird API pipe
    """
    try:
        if params is None:
            params = {}

        # Add token to parameters
        params['token'] = TINYBIRD_TOKEN

        # Make request to Tinybird API
        response = requests.get(TINYBIRD_PIPE_URL, params=params)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        return {"error": f"Failed to query Tinybird API: {str(e)}"}

def get_sample_data_for_suggestions():
    """
    Get a small sample of data specifically for generating suggestions
    Uses limited records to avoid token limits
    """
    try:
        params = {'limit': '50'}  # Only get 50 records for suggestions
        response = requests.get(TINYBIRD_PIPE_URL, params={'token': TINYBIRD_TOKEN, 'limit': '50'})
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": f"Failed to get sample data: {str(e)}"}

def has_zero_results(question, data):
    """
    Check if a question would likely return zero results based on the data structure
    """
    try:        # Convert data to string for analysis
        data_str = json.dumps(data).lower()

        # Keywords that might indicate questions with no data
        no_data_keywords = [
            'tomorrow', 'next week', 'next month', 'next year', 'future',
            'prediction', 'forecast', 'will be', 'going to', 'upcoming',
            '2025', '2026', '2027', '2028', '2029', '2030',  # Future years
            'before 2020', 'before 2019', 'before 2018',  # Too far in the past
            'specific person', 'individual', 'by name', 'personal information',
            'email', 'phone number', 'address', 'contact details'
        ]

        # Check if question contains keywords that would return no data
        question_lower = question.lower()
        for keyword in no_data_keywords:
            if keyword in question_lower:
                return True

        # Check if the data is empty or has very limited data
        if isinstance(data, dict) and 'data' in data:
            if len(data['data']) == 0:
                return True
            # Check if specific data points mentioned in question might be missing
            if 'age' in question_lower and 'age' not in data_str:
                return True
            if ('gender' in question_lower or 'female' in question_lower or 'male' in question_lower) and 'gender' not in data_str:
                return True
            if 'time' in question_lower and 'timestamp' not in data_str and 'time' not in data_str:
                return True

        return False

    except Exception:
        # If we can't determine, assume it might have results
        return False

def clean_gemini_response(response_text):
    """
    Clean the Gemini response by removing table names and other technical references
    """
    # Common table names and abbreviations to remove
    table_patterns = [
        r'\bpc\.', r'\bagd\.', r'\bgd\.', r'\bpctd\.',  # Table prefixes
        r'\bpeople_counting\.', r'\bage_group_details\.', r'\bgender_details\.', r'\bpeople_counting_trend_details\.',  # Full table names
        r'\(pc\)', r'\(agd\)', r'\(gd\)', r'\(pctd\)',  # Table names in parentheses
        r'\[.*?\]',  # Any text in square brackets (often technical references)
        r'\b(table|column|field|join|query|sql)\b',  # Technical database terms
    ]

    # Remove table references
    for pattern in table_patterns:
        response_text = re.sub(pattern, '', response_text, flags=re.IGNORECASE)

    # Clean up extra spaces and punctuation
    response_text = re.sub(r'\s+', ' ', response_text)  # Multiple spaces to single space
    response_text = re.sub(r'\s([.,!?;:])', r'\1', response_text)  # Space before punctuation
    response_text = re.sub(r'\.\.+', '.', response_text)  # Multiple dots to single dot
    response_text = re.sub(r',\s*,', ',', response_text)  # Multiple commas to single comma

    # Capitalize first letter and ensure proper ending
    response_text = response_text.strip()
    if response_text and not response_text.endswith(('.', '!', '?')):
        response_text += '.'

    return response_text

def chunk_data_for_analysis(data, max_chunk_size=500000):
    """
    Split large data into manageable chunks for Gemini analysis
    """
    if not isinstance(data, dict) or 'data' not in data:
        return [data]

    data_items = data['data']
    if not data_items:
        return [data]

    # Convert to JSON string to estimate size
    data_json = json.dumps(data_items)

    # If data is small enough, return as single chunk
    if len(data_json) <= max_chunk_size:
        return [data]

    # Calculate chunk size based on number of items
    avg_item_size = len(data_json) / len(data_items)
    items_per_chunk = max(1, int(max_chunk_size / avg_item_size))

    chunks = []
    for i in range(0, len(data_items), items_per_chunk):
        chunk = {
            'meta': data.get('meta', []),
            'data': data_items[i:i + items_per_chunk],
            'rows': len(data_items[i:i + items_per_chunk]),
            'statistics': data.get('statistics', {})
        }
        chunks.append(chunk)

    return chunks
def analyze_data_with_gemini_stream(question, data):
    """
    Use Gemini to analyze the data and stream the response in real-time
    Generator function that yields chunks of text as they're generated
    """
    try:
        # First, check if we need to chunk the data
        data_chunks = chunk_data_for_analysis(data)

        if len(data_chunks) == 1:
            # Single chunk - process normally with streaming
            prompt = f"""
            Based on the following data from our people counting system, please answer this question: {question}

            Data: {json.dumps(data_chunks[0], indent=2)}

            Please provide a clear, concise answer focusing on the key insights.
            If the data doesn't contain information to answer the question, politely state that.

            FORMATTING RULES:
            1. Use markdown formatting for emphasis:
               - Use **text** for bold/important points
               - Start new sections with **Section Title:** on its own line
               - Use bullet points with * for lists
            2. Use clear paragraph breaks for readability
            3. Structure your response with clear sections when appropriate
            
            IMPORTANT: In your response, use only plain English without any technical references,
            table names, or column names. Focus on the business insights, not the technical implementation.
            """

            # Use Gemini's streaming API
            response = model.generate_content(prompt, stream=True)
            
            for chunk in response:
                if chunk.text:
                    # Don't clean too aggressively during streaming - preserve markdown
                    yield chunk.text

        else:
            # Multiple chunks - process each and combine results with streaming
            chunk_answers = []

            for i, chunk in enumerate(data_chunks):
                chunk_prompt = f"""
                Based on this partial dataset from our people counting system (chunk {i+1}/{len(data_chunks)}),
                please analyze this specific question: {question}

                Partial Data: {json.dumps(chunk, indent=2)}

                Provide key insights or observations from this data chunk that are relevant to the question.
                Focus on patterns, trends, or statistics that help answer the question.
                
                FORMATTING RULES:
                1. Use markdown formatting for emphasis:
                   - Use **text** for bold/important points
                   - Use bullet points with * for lists
                2. Use clear paragraph breaks
                
                IMPORTANT: If this chunk doesn't contain relevant information, simply state "No relevant data in this chunk".
                Use only plain English without technical references.
                """

                try:
                    chunk_response = model.generate_content(chunk_prompt)
                    chunk_answer = chunk_response.text

                    # Only include chunks with meaningful data
                    if "no relevant data" not in chunk_answer.lower():
                        chunk_answers.append(chunk_answer)

                except Exception as chunk_error:
                    # Continue with other chunks if one fails
                    continue

            # If no chunks produced meaningful answers
            if not chunk_answers:
                yield "I don't have enough relevant data to answer this question based on the current dataset."
                return

            # Combine insights from all chunks
            if len(chunk_answers) == 1:
                for char in chunk_answers[0]:
                    yield char
                return

            # Use Gemini to synthesize the chunk answers with streaming
            synthesis_prompt = f"""
            Question: {question}

            I have analyzed this question across multiple data chunks and found these insights:

            {chr(10).join([f"Chunk {i+1}: {answer}" for i, answer in enumerate(chunk_answers)])}

            Please synthesize these insights into a single, coherent answer to the original question.
            Focus on the most important patterns and provide a clear, concise response.
            
            FORMATTING RULES:
            1. Use markdown formatting for emphasis:
               - Use **text** for bold/important points
               - Start sections with **Section Title:** on new lines
               - Use bullet points with * for lists
            2. Use clear paragraph breaks for readability

            IMPORTANT: Use only plain English without technical references. Provide a business-focused answer.
            """

            try:
                synthesis_response = model.generate_content(synthesis_prompt, stream=True)
                for chunk in synthesis_response:
                    if chunk.text:
                        yield chunk.text
            except Exception:
                # Fallback: return the first meaningful chunk answer
                for char in chunk_answers[0]:
                    yield char

    except Exception as e:
        yield f"Error analyzing data with Gemini: {str(e)}"
def generate_chart_with_gemini(question, data, answer):
    """
    Use Gemini to generate an HTML chart based on the question, data, and answer
    Enhanced to support all chart types: bar, column, line, pie, doughnut, area
    """
    try:
        # First check if we have enough data to generate a meaningful chart
        if has_zero_results(question, data):
            return "<div class='chart-error'>No data available to generate a chart for this question.</div>"

        # Use a smaller data sample for chart generation to avoid token limits
        chart_data_sample = get_chart_data_sample(data)

        prompt = f"""
        Based on the following question, data sample, and answer, generate an HTML chart using Chart.js.

        Question: {question}
        Data Sample: {json.dumps(chart_data_sample, indent=2)}
        Answer: {answer}

        Create an appropriate chart that visualizes the key insights from the answer.
        Choose the BEST chart type based on the data and question:
        
        - BAR CHART: For comparing categories (horizontal bars)
        - COLUMN CHART: For comparing values across categories (vertical bars, use 'bar' type with indexAxis: 'x')
        - LINE CHART: For trends over time or continuous data
        - PIE CHART: For showing parts of a whole (percentage/proportion data)
        - DOUGHNUT CHART: Similar to pie but with center hole (use 'doughnut' type)
        - AREA CHART: For showing volume/magnitude over time (line chart with filled area)
        
        The chart should be responsive and well-styled.

        IMPORTANT REQUIREMENTS:
        1. Return ONLY the HTML code with embedded JavaScript for the chart.
        2. Use Chart.js version 3.x or 4.x syntax - DO NOT include the Chart.js CDN script
        3. Use a UNIQUE canvas ID: "chart_" + random 6-digit number
        4. Make the chart RESPONSIVE with proper configuration
        5. Use appropriate colors and labels that match the question
        6. Add a meaningful title that summarizes the insight
        7. Choose the chart type that BEST represents the data
        8. Include proper chart options for responsiveness
        9. Do NOT include any explanations outside the HTML code
        10. Ensure canvas has proper wrapper div with responsive styles

        CRITICAL CHART.JS SYNTAX:
        - For responsive charts, use: responsive: true, maintainAspectRatio: true
        - For bar charts (vertical): type: 'bar', no indexAxis needed (default is 'x')
        - For horizontal bars: type: 'bar', options: {{ indexAxis: 'y' }}
        - For doughnut: type: 'doughnut'
        - For area charts: type: 'line' with fill: true in dataset
        - Always wrap canvas in a container div with proper styling

        Example structure (adapt based on chosen chart type):
        <div style="width: 100%; max-width: 700px; margin: 20px auto; padding: 20px; background: white; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
            <canvas id="chart_XXXXXX" style="width: 100%; height: 400px;"></canvas>
        </div>
        <script>
        (function() {{
            const ctx = document.getElementById('chart_XXXXXX');
            if (!ctx) return;
            
            // Destroy existing chart if any
            const existingChart = Chart.getChart(ctx);
            if (existingChart) existingChart.destroy();
            
            new Chart(ctx, {{
                type: 'bar', // or 'line', 'pie', 'doughnut'
                data: {{
                    labels: ['Label1', 'Label2'],
                    datasets: [{{
                        label: 'Dataset Label',
                        data: [value1, value2],
                        backgroundColor: ['#4f46e5', '#8b5cf6'],
                        borderColor: ['#4338ca', '#7c3aed'],
                        borderWidth: 2,
                        fill: false // set to true for area charts
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {{
                        title: {{
                            display: true,
                            text: 'Chart Title',
                            font: {{ size: 16, weight: 'bold' }}
                        }},
                        legend: {{
                            display: true,
                            position: 'top'
                        }},
                        tooltip: {{
                            enabled: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }})();
        </script>
        
        Generate a complete, working chart that displays properly in the browser.
        """

        response = model.generate_content(prompt)

        # Extract the HTML code from the response
        html_code = response.text.strip()

        # Clean up the response to ensure it's valid HTML
        if html_code.startswith('```html'):
            html_code = html_code[7:]
        if html_code.endswith('```'):
            html_code = html_code[:-3]
        if html_code.startswith('```'):
            html_code = html_code[3:]
        if html_code.endswith('```'):
            html_code = html_code[:-3]

        html_code = html_code.strip()

        # Remove any Chart.js CDN script tags
        html_code = re.sub(r'<script[^>]*src=["\'][^"\']*chart[^"\']*["\'][^>]*>.*?</script>', '', html_code, flags=re.IGNORECASE | re.DOTALL)

        # Ensure we have a proper canvas element
        if '<canvas' not in html_code:
            html_code = '''
            <div style="width: 100%; max-width: 700px; margin: 20px auto; padding: 20px; background: white; border-radius: 12px;">
                <p style="text-align: center; color: #666;">Unable to generate chart visualization</p>
            </div>
            '''

        return html_code

    except Exception as e:
        return f"<div class='chart-error'>Error generating chart: {str(e)}</div>"

def get_chart_data_sample(data, max_samples=100):
    """
    Get a sample of data for chart generation to avoid token limits
    """
    if not isinstance(data, dict) or 'data' not in data:
        return data

    if len(data['data']) <= max_samples:
        return data

    # Create a sampled version
    sampled_data = data.copy()
    sampled_data['data'] = data['data'][:max_samples]
    sampled_data['rows'] = len(sampled_data['data'])
    sampled_data['sampled'] = True
    sampled_data['total_original_rows'] = len(data['data'])

    return sampled_data
