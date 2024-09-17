from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import datetime
from datetime import datetime
import os
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import time
import re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from flask_mail import Mail, Message
from groq import Groq
load_dotenv()



app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mydatabase'
mongo = PyMongo(app)

app.secret_key = os.urandom(24)  # Required for flashing messages

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = ''
app.config['MAIL_PASSWORD'] = ''
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)





@app.route('/')
def home():
    return render_template('login.html')


@app.route('/change-role', methods=['GET'])
def change_role():
    email = request.args.get('email')
    
    if not email:
        return jsonify({'error': 'Email parameter is missing'}), 400

    # Define role mapping based on email
    role_mapping = {
        'haseeb@gmail.com': True
    }

    new_role = role_mapping.get(email)
    
    if new_role is None:
        return jsonify({'error': 'No role defined for this email'}), 404

    # Update the user's role
    result = mongo.db.users.update_one(
        {'username': email},
        {'$set': {'is_admin': new_role}}
    )

    if result.matched_count == 0:
        return jsonify({'error': 'User not found'}), 404

    return jsonify({'message': 'User role updated to admin successfully'})

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = mongo.db.users.find_one({'username': username})
    if user and check_password_hash(user['password'], password):
        session['username'] = username # Store the username in the session
        session['is_admin'] = user.get('is_admin', False)  # Check if user is admin
        
        # Initialize or clear session-specific message history
        app.config['messages'] = []
        
        if session['is_admin']:
            return redirect(url_for('admin'))  # Redirect to the admin page if the user is an admin
        return redirect(url_for('index')) # Redirect to the chatbot page
    else:
        flash('Invalid username or password.')
        return redirect(url_for('home'))

@app.route('/logout')
def logout():
    session.pop('messages', None)  # Remove the chat history from the session
    session.pop('username', None)  # Remove the username from the session
    session.pop('is_admin', None)
    flash('You have been logged out.')
    return redirect(url_for('home'))


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check if the username already exists in the database
        existing_user = mongo.db.users.find_one({'username': username})
        
        if existing_user:
            # If the username already exists, flash a message and redirect to signup page
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('signup'))
        # If username does not exist, proceed to create a new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        mongo.db.users.insert_one({'username': username, 'password': hashed_password})
        
        # Also create an entry in the attendance collection
        mongo.db.attendance.insert_one({
            'user_id': username,
            'total availed sick leaves': 0,
            'total availed annual leaves': 0,
            'total availed wfh leaves': 0,
            'total availed casual leaves': 0,
            'total availed extra leaves': 0,
            'total allocated sick leaves': 10,
            'total allocated annual leaves': 10,
            'total allocated wfh leaves': 2,
            'total allocated casual leaves': 10,
            'total allocated extra leaves': 10,
            'total remaining annual leaves': 10,
            'total remaining casual leaves': 10,
            'total remaining extra leaves': 10,
            'total remaining sick leaves': 10,
            'total remaining wfh leaves': 2,
            'status': 'probation'
        })
        
        flash('Signup successful! Please log in.')
        return redirect(url_for('home'))
    return render_template('signup.html')

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        username = request.form['username']
        user = mongo.db.users.find_one({'username': username})
        if user:
            # Generate a token and store it
            token = str(uuid.uuid4())
            mongo.db.password_reset_tokens.insert_one({
                'username': username,
                'token': token,
                'expires_at': datetime.datetime.now() + datetime.timedelta(hours=1)  # 1 hour expiration
            })
            # Generate a reset URL
            reset_url = url_for('reset_password', token=token, _external=True)
            flash(f'Click the link to reset your password: <a href="{reset_url}">{reset_url}</a>', 'info')
        else:
            flash('Username not found.')
        return redirect(url_for('forgot_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    if request.method == 'POST':
        new_password = request.form['password']
        reset_token = mongo.db.password_reset_tokens.find_one({'token': token})
        if reset_token and reset_token['expires_at'] > datetime.datetime.now():
            username = reset_token['username']
            hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
            mongo.db.users.update_one({'username': username}, {'$set': {'password': hashed_password}})
            mongo.db.password_reset_tokens.delete_one({'token': token})
            flash('Password has been reset successfully! You can now log in.')
            return redirect(url_for('home'))
        else:
            flash('Invalid or expired token.')
    return render_template('reset_password.html', token=token)

# Directly set environment variables
os.environ['GROQ_API_KEY'] = ''
os.environ['GOOGLE_API_KEY'] = ''

# Access the keys
groq_api_key = os.getenv("GROQ_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")

print(f"GROQ_API_KEY: {groq_api_key}")
print(f"GOOGLE_API_KEY: {google_api_key}")

llm = ChatGroq(groq_api_key = groq_api_key,
               model_name = "Llama3-8b-8192")
prompt = ChatPromptTemplate.from_template(
    """You are the HR representative.  Provide a friendly and relevant response for these questions.
    If the question is specific to the provided context, answer it based on the context only. Please provide the most accurate response based on the question. If an answer cannot be found in the context, provide the closest related answer.
    if username is not admin and trying to asker other user data, then provide the response that you are not allowed to access this information.

Current User: {username}
User Details:
<context> {context} <context>
<user_attendance>{attendance_data}<user_attendance>
Questions: {input}
is_admin: {admin}
"""
)

# Initialize conversation memory
memory = ConversationBufferMemory()

# Define a function for handling common questions
def handle_common_questions(question):
    common_responses = {
        "hi": "Hello! How can I assist you today?",
        "hello": "Hi there! What can I help you with?",
        # Add more common questions and responses here
    }

    # Handle common questions with predefined responses
    if question.lower() in common_responses:
        return common_responses.get(question.lower())

def vector_embedding():
    if "vectors" not in app.config:
        app.config['embeddings'] = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        app.config['loader'] = PyPDFDirectoryLoader("./data")
        app.config['docs'] = app.config['loader'].load()
        app.config['text_splitter'] = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)
        app.config['final_documents'] = app.config['text_splitter'].split_documents(app.config['docs'][:20])
        app.config['vectors'] = FAISS.from_documents(app.config['final_documents'],app.config['embeddings'])

def format_response(response):
    # Format the response with HTML line breaks
    formatted_response = response.replace("\n", "<br>")
    return formatted_response



def check_user_permissions(query, current_username):
    # Extract the name from the query
    extracted_names = name_extractor(query)
    
    # Check if the extracted name matches the current session username
    if current_username in extracted_names:
        return "you are allowed to access this information"
    else:
        return False


db = mongo.db
collection = db['attendance']

def name_extractor(text):
    """
    Extracts potential user identifiers from text using NLTK.
    Combines Named Entity Recognition (NER) with additional regex for better coverage.
    """
    potential_users = []

    # Tokenization and Part-of-Speech (POS) Tagging
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)

    # Named Entity Recognition (NER)
    named_entities = ne_chunk(tagged_tokens)
    print("Named Entities:", named_entities)  # Debugging output
    for entity in named_entities:
        if hasattr(entity, 'label') and entity.label() == 'PERSON':
            user = ' '.join([word for word, tag in entity.leaves()])
            potential_users.append(user)

    # Additional Pattern Matching for Capitalized Names
    if not potential_users:
        # Pattern to match sequences of capitalized words
        pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)+\b'
        matches = re.findall(pattern, text)
        potential_users.extend(matches)
    
    # Enhance pattern to catch single capitalized names
    if not potential_users:
        pattern_single = r'\b[A-Z][a-z]+\b'
        single_matches = re.findall(pattern_single, text)
        potential_users.extend(single_matches)

    # Deduplicate and filter out invalid names (only full names with two or more words)
    filtered_users = set()
    for user in potential_users:
        # Filter out names that are not full names
        if len(user.split()) >= 2:  # Allow names with two or more words
            filtered_users.add(user)
        elif len(user.split()) == 1:  # Allow single capitalized names if they seem like valid names
            filtered_users.add(user)
    
    return list(filtered_users)


def fetch_data(names):
    if not names:
        return "No names provided."
    
    # Initialize an empty list to hold the fetched data
    data = []
    
    for name in names:
        # Construct a regex pattern for flexible matching
        regex = re.compile(f'{re.escape(name)}', re.IGNORECASE)
        
        # Fetch attendance records from MongoDB
        results = mongo.db.attendance.find({'user_id': regex})
        
        # Process and format each record
        for result in results:
            formatted_result = (
                f"Name: {result.get('user_id', 'N/A')}, "
                f"Status: {result.get('status', 'N/A')}, "
                f"Total allocated annual leaves: {result.get('total allocated annual leaves', 'N/A')}, "
                f"Total allocated casual leaves: {result.get('total allocated casual leaves', 'N/A')}, "
                f"Total allocated extra leaves: {result.get('total allocated extra leaves', 'N/A')}, "
                f"Total allocated sick leaves: {result.get('total allocated sick leaves', 'N/A')}, "
                f"Total allocated WFH leaves: {result.get('total allocated wfh leaves', 'N/A')}, "
                f"Total availed annual leaves: {result.get('total availed annual leaves', 'N/A')}, "
                f"Total availed casual leaves: {result.get('total availed casual leaves', 'N/A')}, "
                f"Total availed extra leaves: {result.get('total availed extra leaves', 'N/A')}, "
                f"Total availed sick leavess: {result.get('total availed sick leaves', 'N/A')}, "
                f"Total availed WFH leaves: {result.get('total availed wfh leaves', 'N/A')}, "
                f"Total remaining annual leaves: {result.get('total remaining annual leaves', 'N/A')}, "
                f"Total remaining casual leaves: {result.get('total remaining casual leaves', 'N/A')}, "
                f"Total remaining extra leaves: {result.get('total remaining extra leaves', 'N/A')}, "
                f"Total remaining sick leaves: {result.get('total remaining sick leaves', 'N/A')}, "
                f"Total remaining WFH leaves: {result.get('total remaining wfh leaves', 'N/A')}"
            )
            data.append(formatted_result)
    
    # Return the formatted data or a message if no records were found
    return " ".join(data) if data else "No relevant information found in the database."



def fetch_employee_data(name):
    if not name:
        return "No names provided."

    # Initialize an empty list to hold the fetched data
    data = []
    name=str(name)

        # Aggregation pipeline
    pipeline = [
    {'$match': {'user_id': {'$regex': f"{name}", '$options': 'i'}}},
    {'$group': {  
        '_id': '$user_id',
        'status': {'$first': '$status'},
        'total allocated annual leaves': {'$sum': '$total allocated annual leaves'},
        'total allocated casual leaves': {'$sum': '$total allocated casual leaves'},
        'total allocated extra leaves': {'$sum': '$total allocated extra leaves'},
        'total allocated sick leaves': {'$sum': '$total allocated sick leaves'},
        'total allocated wfh leaves': {'$sum': '$total allocated wfh leaves'},
        'total availed annual leaves': {'$sum': '$total availed annual leaves'},
        'total availed casual leaves': {'$sum': '$total availed casual leaves'},
        'total availed extra leaves': {'$sum': '$total availed extra leaves'},
        'total availed sick leaves': {'$sum': '$total availed sick leaves'},
        'total availed wfh leaves': {'$sum': '$total availed wfh leaves'},
        'total remaining annual leaves': {'$sum': '$total remaining annual leaves'},
        'total remaining casual leaves': {'$sum': '$total remaining casual leaves'},
        'total remaining extra leaves': {'$sum': '$total remaining extra leaves'},
        'total remaining sick leaves': {'$sum': '$total remaining sick leaves'},
        'total remaining wfh leaves': {'$sum': '$total remaining wfh leaves'}
    }},
    {'$project': {
        '_id': 0,
        'Name': '$_id',
        'status': 1,
        'Total allocated annual leaves': '$total allocated annual leaves',
        'Total allocated casual leaves': '$total allocated casual leaves',
        'Total allocated extra leaves': '$total allocated extra leaves',
        'Total allocated sick leaves': '$total allocated sick leaves',
        'Total allocated WFH leaves': '$total allocated wfh leaves',
        'Total availed annual leaves': '$total availed annual leaves',
        'Total availed casual leaves': '$total availed casual leaves',
        'Total availed extra leaves': '$total availed extra leaves',
        'Total availed sick leaves': '$total availed sick leaves',
        'Total availed WFH leaves': '$total availed wfh leaves',
        'Total remaining annual leaves': '$total remaining annual leaves',
        'Total remaining casual leaves': '$total remaining casual leaves',
        'Total remaining extra leaves': '$total remaining extra leaves',
        'Total remaining sick leaves': '$total remaining sick leaves',
        'Total remaining WFH leaves': '$total remaining wfh leaves'
    }}
]

        # Execute the aggregation pipeline
    results = mongo.db.attendance.aggregate(pipeline)

        # Process and format each record
    for result in results:
        formatted_result = (
            f"Name: {result.get('Name', 'N/A')}, "
            f"Status: {result.get('status', 'N/A')}, "
            f"Total allocated annual leaves: {result.get('Total allocated annual leaves', 'N/A')}, "
            f"Total allocated casual leaves: {result.get('Total allocated casual leaves', 'N/A')}, "
            f"Total allocated extra leaves: {result.get('Total allocated extra leaves', 'N/A')}, "
            f"Total allocated sick leaves: {result.get('Total allocated sick leaves', 'N/A')}, "
            f"Total allocated WFH leaves: {result.get('Total allocated WFH leaves', 'N/A')}, "
            f"Total availed annual leaves: {result.get('Total availed annual leaves', 'N/A')}, "
            f"Total availed casual leaves: {result.get('Total availed casual leaves', 'N/A')}, "
            f"Total availed extra leaves: {result.get('Total availed extra leaves', 'N/A')}, "
            f"Total availed sick leaves: {result.get('Total availed sick leaves', 'N/A')},"
            f"Total availed WFH leaves: {result.get('Total availed WFH leaves', 'N/A')}, "
            f"Total remaining annual leaves: {result.get('Total remaining annual leaves', 'N/A')}, "
            f"Total remaining casual leaves: {result.get('Total remaining casual leaves', 'N/A')}, "
            f"Total remaining extra leaves: {result.get('Total remaining extra leaves', 'N/A')}, "
            f"Total remaining sick leaves: {result.get('Total remaining sick leaves', 'N/A')},"
            f"Total remaining WFH leaves: {result.get('Total remaining WFH leaves', 'N/A')}"
        )
        data.append(formatted_result)

    # Return the formatted data or a message if no records were found
    return " ".join(data) if data else "No relevant information found in the database."


def send_email(recipient, subject, body):
    msg = Message(subject, sender='', recipients=[recipient])
    msg.body = body
    try:
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Error sending email: {str(e)}"



# Set the Groq API Key
GROQ_API_KEY = ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

client = Groq()

def process_chat(name: str, user_prompt: str) -> str:
    """
    Function to process a single chat interaction.

    Parameters:
        name (str): The name of the user.
        user_prompt (str): The user's input message.

    Returns:
        str: The assistant's response.
    """
    # name of user
    name = name
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Regular expression pattern to match email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@gmail\.com\b'
    
    # Search for the email address in the query
    email_match = re.search(email_pattern, user_prompt)
    salutation = None
    if email_match:
        email = email_match.group()
        salutation = email.split('@')[0]
        print(f"Salutation extracted: {salutation}")
    else:
        salutation = "None"  # Default salutation if no email found
    
    # Regular expression pattern to match text after 'about'
    about_pattern = r'\babout\b\s*(.*)'
    about_match = re.search(about_pattern, user_prompt, re.IGNORECASE)
    about = None
    if about_match:
        about = about_match.group(1)
        print(f"Text after 'about': {about}")
    else:
        about = "No specific topic provided."
    
    # Define the conversation context, include the date in the system message content
    messages = [
        {"role": "system", "content": f"You are an expert email body writer who can write all queries related to email body without a subject but nothing else. You will only write answers related to email body. Today's date is {current_date}. My name is {name}. Name for salutation is {salutation}. The topic is '{about}'."},
        {"role": "user", "content": user_prompt}
    ]

    # Generate response
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages
    )
    assistant_response = response.choices[0].message.content

    return assistant_response




@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    vector_embedding()

    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        username = session.get('username')
        is_admin = session.get('is_admin', False)

        print(f"Received question: {question}")
        print(f"Current user: {username}, Admin status: {is_admin}")

        # Check if the question is an email writing request
        if "write an email to" in question.lower():
            try:
                # Extract recipient and topic from the question
                recipient = re.search(r'write an email to ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', question).group(1)
                topic = re.search(r'about (.*)', question).group(1)

                # Validate the recipient email address
                if not recipient or not topic:
                    raise ValueError("Recipient or topic is missing.")
                # Generate email body using process_chat function
                email_body = process_chat(username, question)

                # Ask for confirmation
                response = f"Do you want to send an email to {recipient} with subject '{topic}' and body '{email_body}'? (yes/no)"
                session['email_confirmation'] = {
                    'recipient': recipient,
                    'subject': topic,
                    'body': email_body
                }
            except Exception as e:
                response = f"Error parsing email request: {str(e)}"

        # Check if the chatbot is waiting for confirmation
        elif 'email_confirmation' in session:
            confirmation = question.strip().lower()
            if confirmation == 'yes':
                email_data = session.pop('email_confirmation')
                email_response = send_email(email_data['recipient'], email_data['subject'], email_data['body'])
                response = email_response
            elif confirmation == 'no':
                session.pop('email_confirmation')
                response = "Email sending cancelled."
            else:
                response = "Please confirm by typing 'yes' or 'no'."
            
        else:
            common_response = handle_common_questions(question)
            if common_response:
                response = common_response
            else:
                if is_admin:
                    name = name_extractor(question)
                    name = name[0] if name else None
                    attendance_data = fetch_employee_data(name)
                    print(f"Attendance data: {attendance_data}")
                    print(f"Extracted names: {name}")
                    prompt_input = {
                        'username': username,
                        'name': name,
                        'context': f"{attendance_data}\n{memory.buffer}",
                        'input': question,
                        'attendance_data': attendance_data,  # Ensure this is included
                        "admin": is_admin
                    }
                else:
                    user_attendance = mongo.db.attendance.find_one({'user_id': username})
                    print(type(user_attendance))
                    user_attendance_data = fetch_data([username]) if user_attendance else "No attendance data found for this user."
                    prompt_input = {
                        'username': username,
                        'context': f"{username}\n{memory.buffer}",
                        'user_attendance': user_attendance_data,  # Add this line
                        'input': question,
                        'attendance_data': user_attendance_data,  # Ensure this is included
                        "admin": is_admin
                    }

                print(f"Prompt input: {prompt_input}")

                document_chain = create_stuff_documents_chain(llm, prompt)
                retriever = app.config["vectors"].as_retriever()
                retrieval_chain = create_retrieval_chain(retriever, document_chain)

                start = time.process_time()
                response = retrieval_chain.invoke(prompt_input)['answer']
                response_time = time.process_time() - start
                response += f" (Response Time: {response_time:.2f} seconds)"

                print(f"Response: {response}")

                memory.save_context({"input": question}, {"response": response})

        app.config['messages'].append((question, response))
    return render_template('index.html', messages=app.config.get('messages', []))






@app.route('/admin', methods=['GET', 'POST'])
def admin():
    attendance_record = None
    attendance_records = []

    if request.method == 'POST':
        if 'view' in request.form:
            user_id = request.form.get('user_id')
            attendance_record = mongo.db.attendance.find_one({'user_id': user_id})
        elif 'add_update' in request.form:
            user_id = request.form.get('user_id')
            update_data = {
                'total allocated sick leaves': float(request.form.get('total_allocated_sick_leaves', 0)),
                'total availed sick leaves': float(request.form.get('total_availed_sick_leaves', 0)),
                'total allocated annual leaves': float(request.form.get('total_allocated_annual_leaves', 0)),
                'total availed annual leaves': float(request.form.get('total_availed_annual_leaves', 0)),
                'total allocated wfh leaves': float(request.form.get('total_allocated_wfh_leaves', 0)),
                'total availed wfh leaves': float(request.form.get('total_availed_wfh_leaves', 0)),
                'total allocated casual leaves': float(request.form.get('total_allocated_casual_leaves', 0)),
                'total availed casual leaves': float(request.form.get('total_availed_casual_leaves', 0)),
                'total allocated extra leaves': float(request.form.get('total_allocated_extra_leaves', 0)),
                'total availed extra leaves': float(request.form.get('total_availed_extra_leaves', 0)),
                'status': request.form.get('status', 'Probation')
            }

            # Calculate remaining leaves
            update_data['total remaining annual leaves'] = update_data['total allocated annual leaves'] - update_data['total availed annual leaves']
            update_data['total remaining casual leaves'] = update_data['total allocated casual leaves'] - update_data['total availed casual leaves']
            update_data['total remaining extra leaves'] = update_data['total allocated extra leaves'] - update_data['total availed extra leaves']
            update_data['total remaining sick leaves'] = update_data['total allocated sick leaves'] - update_data['total availed sick leaves']
            update_data['total remaining wfh leaves'] = update_data['total allocated wfh leaves'] - update_data['total availed wfh leaves']

            mongo.db.attendance.update_one({'user_id': user_id}, {'$set': update_data}, upsert=True)

        # Fetch all attendance records
        attendance_records = list(mongo.db.attendance.find({}))

    return render_template('admin.html', attendance_record=attendance_record, attendance_records=attendance_records)





if __name__ == '__main__':
    app.run()
