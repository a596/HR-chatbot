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
from langchain_community.document_loaders.csv_loader import CSVLoader
from dotenv import load_dotenv
import time
import re
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from flask_mail import Mail, Message
from groq import Groq
import gspread
import pandas as pd
from google.oauth2.service_account import Credentials
load_dotenv()


# Initialize conversation memory
memory = ConversationBufferMemory()
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['MONGO_URI'] = 'mongodb://localhost:27017/mydatabase'
mongo = PyMongo(app)

app.secret_key = os.urandom(24)  # Required for flashing messages

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'ethicalgan@gmail.com'
app.config['MAIL_PASSWORD'] = 'rehg hjfx tauh zrof'
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
        
        username = reset_token['username']
        hashed_password = generate_password_hash(new_password, method='pbkdf2:sha256')
        mongo.db.users.update_one({'username': username}, {'$set': {'password': hashed_password}})
        mongo.db.password_reset_tokens.delete_one({'token': token})
        flash('Password has been reset successfully! You can now log in.')
        return redirect(url_for('home'))
        
    return render_template('reset_password.html', token=token)

# Directly set environment variables
os.environ['GROQ_API_KEY'] = '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'
os.environ['GOOGLE_API_KEY'] = 'AIzaSyAke4hturZTQtCvS0SLA00t0rD5MJifhW4'

# Access the keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

print(f"GROQ_API_KEY: {GROQ_API_KEY}")
print(f"GOOGLE_API_KEY: {GOOGLE_API_KEY}")


    
def send_email(recipient, subject, body):
    msg = Message(subject, sender='ethicalgan@gmail.com', recipients=[recipient])
    msg.body = body
    try:
        mail.send(msg)
        return "Email sent successfully!"
    except Exception as e:
        return f"Error sending email: {str(e)}"
    
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


# Initialize the model and prompts
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
answer_prompt = ChatPromptTemplate.from_template(
    """
    Answere the general Question gracefully. Don't let anyone know you are a bot.
    You are the HR representative. Provide a direct, clear, and relevant response to the questions based on the provided context only. If the user tries ask to other person attendance info then you can say that you are not authorized to provide that information.
    username : {username}
    <context>
    {context}
    </context>
    Questions: {input}"""
)

source_selection_prompt = ChatPromptTemplate.from_template(
    """Given the following question, determine whether it's more likely to be answered using PDF documents or CSV data.
    Respond with either "pdf" or "csv".
    
    Question: {input}
    
    Your response (pdf/csv):"""
)

import logging

def create_vector_store(docs, embeddings):
    if not docs:
        print("No documents to process.")
        return None
    
    '''text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)#1200
    split_docs = text_splitter.split_documents(docs[:100])#20
    print(f"Number of documents after splitting: {len(split_docs)}")'''
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=20)#1200
    split_docs = text_splitter.split_documents(docs[:20])#20
    print(f"Number of documents after splitting: {len(split_docs)}")
    
    if not split_docs:
        print("No documents after splitting.")
        return None
    
    try:
        print(f"Creating FAISS index with {len(split_docs)} documents...")
        faiss_index = FAISS.from_documents(split_docs, embeddings)
        print("FAISS index created successfully.")
        return faiss_index
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None



def create_pdf_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    pdf_loader = PyPDFDirectoryLoader("./data")
    pdf_docs = pdf_loader.load()
    print(f"Number of PDF documents loaded: {len(pdf_docs)}")
    pdf_vector_store = create_vector_store(pdf_docs, embeddings)
    print(f"PDF vector store creation status: {'Success' if pdf_vector_store else 'Failed'}")
    return pdf_vector_store

def create_csv_vector_store(filepath):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    csv_loader = CSVLoader(filepath)
    csv_docs = csv_loader.load()
    print(f"Number of CSV documents loaded: {len(csv_docs)}")
    csv_vector_store = create_vector_store(csv_docs, embeddings)
    print(f"CSV vector store creation status: {'Success' if csv_vector_store else 'Failed'}")
    return csv_vector_store



def create_agent(vector_store):
    if vector_store is None:
        return None
    document_chain = create_stuff_documents_chain(llm, answer_prompt)
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )
    return create_retrieval_chain(retriever, document_chain)

def select_source(question):
    selection_chain = source_selection_prompt | llm
    response = selection_chain.invoke({'input': question})
    return response.content.strip().lower()


def process_google_sheet_to_csv(sheet_url, service_account_file, csv_file_path):
    SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    credentials = Credentials.from_service_account_file(service_account_file, scopes=SCOPES)
    gc = gspread.authorize(credentials)
    sheet = gc.open_by_url(sheet_url)
    worksheet = sheet.get_worksheet(0)
    data = worksheet.get_all_values()
    df = pd.DataFrame(data)
    df = df.replace(r'^\s*$', pd.NA, regex=True).dropna(axis=1, how='all')
    header1 = df.iloc[0].fillna('')
    header2 = df.iloc[1].fillna('')
    combined_headers = [f"{str(h1).strip()} {str(h2).strip()}" if pd.notna(h1) or pd.notna(h2) else '' for h1, h2 in zip(header1, header2)]
    desired_headers = ["Employee Name", "Status", "Availed Annual Leave", "Availed Casual Leave", "Availed Sick Leave", "Availed WFH Leave", "Availed Extra Leave", "Allocated Annual Leave", "Allocated Casual Leave", "Allocated Sick Leave", "Allocated Extra Leave", "Allocated WFH Leave", "Remaining Annual Leave", "Remaining Casual Leave", "Remaining Sick Leave", "Remaining Extra Leave", "Remaining WFH Leave"]
    
    print(f"Initial columns: {df.columns.tolist()}")
    print(f"Combined headers: {combined_headers}")
    
    if len(desired_headers) == len(df.columns):
        df.columns = desired_headers
    else:
        print(f"Number of desired headers does not match the number of columns.")
    
    df = df.drop([0, 1]).reset_index(drop=True).fillna(0)
    df.to_csv(csv_file_path, index=False)
    print(f"CSV file saved to: {csv_file_path}")
    return csv_file_path



# Example usage
csv_file_path = process_google_sheet_to_csv(
    'https://docs.google.com/spreadsheets/d/1nE5qWj0v3mkrErGcOkR6jyaqQ-9hVpFvUAvNDM3Bvwg/edit?gid=1998366789#gid=1998366789',
    'data/flaskapp-435311-f2c370b6b0e6.json',
    'data/Annual Leave Sheet Qwerty 2024.csv'
)

logging.info("Creating PDF vector store...")
pdf_vector_store = create_pdf_vector_store()
logging.info("Creating CSV vector store...")
csv_vector_store = create_csv_vector_store(csv_file_path)

pdf_chain = create_agent(pdf_vector_store)
csv_chain = create_agent(csv_vector_store)

def process_leave_query(question, csv_file_path):
    df = pd.read_csv(csv_file_path)
    
    print(" Hi, I am here in process leave query")
    
    # Convert question to lowercase for case-insensitive matching
    question_lower = question.lower()
    
    if 'annual leaves' in question_lower and '0' in question_lower or 'total annual leave' in question_lower and '0' in question_lower or 'total remaining annual leave' in question_lower and '0' in question_lower or 'remaining annual leaves' in question_lower and '0' in question_lower:
        result = df[df['Remaining Annual Leave'] == 0]['Employee Name'].tolist()
        print(result)
    elif 'sick leaves' in question_lower and '0' in question_lower or 'total sick leaves' in question_lower and '0' in question_lower or 'total remaining sick leave' in question_lower and '0' in question_lower or 'remaining sick leaves' in question_lower and '0' in question_lower:
        result = df[df['Remaining Sick Leave'] == 0]['Employee Name'].tolist()
        print(result)
    elif 'casual leaves' in question_lower and '0' in question_lower  or 'total casual leaves' in question_lower and '0' in question_lower or 'total remaining casual leave' in question_lower and '0' in question_lower or 'remaining casual leaves' in question_lower and '0' in question_lower:
        result = df[df['Remaining Casual Leave'] == 0]['Employee Name'].tolist()
        print(result)
    elif 'wfh leaves' in question_lower and '0' in question_lower or 'total wfh leaves' in question_lower and '0' in question_lower or 'total remaining wfh leave' in question_lower and '0' in question_lower or 'remaining wfh leaves' in question_lower and '0' in question_lower:
        result = df[df['Remaining WFH Leave'] == 0]['Employee Name'].tolist()
        print(result)
    elif 'extra leaves' in question_lower and '0' in question_lower or 'total extra leaves' in question_lower and '0' in question_lower or 'total remaining extra leave' in question_lower and '0' in question_lower or 'remaining extra leaves' in question_lower and '0' in question_lower:
        result = df[df['Remaining Extra Leave'] == 0]['Employee Name'].tolist()
        print(result)
    else:
        return "I'm sorry, I couldn't understand the specific leave type or condition you're asking about. Could you please rephrase your question?"

    if result:
        return f"The following employees have 0 of the specified leave type: {', '.join(result)}"
    else:
        return "No employees found matching the specified criteria."


@app.route('/chatbot', methods=['GET', 'POST'])
def index():
    if 'messages' not in app.config:
        app.config['messages'] = []

    if request.method == 'POST':
        question = request.form.get('question')
        username = session.get('username')
        is_admin = session.get('is_admin', False)
        chain = None

        print(f"Received question: {question}")
        print(f"Current user: {username}, Admin status: {is_admin}")

        if "write an email to" in question.lower() or "send an email to" in question.lower() or "email" in question.lower():# send email, email
            try:
                recipient = re.search(r'(?:write an email to|send an email to|email to) ([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', question).group(1)
                topic = re.search(r'about (.*)', question).group(1)
                if not recipient or not topic:
                    raise ValueError("Recipient or topic is missing.")
                email_body = process_chat(username, question)
                response = f"Do you want to send an email to {recipient} with subject '{topic}' and body '{email_body}'? (yes/no)"
                session['email_confirmation'] = {
                    'recipient': recipient,
                    'subject': topic,
                    'body': email_body
                }
            except Exception as e:
                response = f"Error parsing email request: {str(e)}"
                print(f"Email request parsing error: {str(e)}")

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
            if is_admin:
                username = username
                print(username)
                
                # Check if it's a leave query
                if any(leave_type in question.lower() for leave_type in ['annual leaves', 'sick leaves', 'casual leaves', 'wfh leaves', 'extra leaves']):
                    response = process_leave_query(question, csv_file_path)
                    app.config['messages'].append((question, response))
                    return render_template('index.html', messages=app.config.get('messages', []))
                else:
                    source = select_source(question)
                    print(f"Selected source: {source}")
                    
                    if source == 'pdf' and pdf_chain:
                        chain = pdf_chain
                    elif source == 'csv' and csv_chain:
                        chain = csv_chain
                    else:
                        print("No valid agents available. Skipping question.")
                        response = answer_prompt.invoke({'input': question, 'context': '', 'username': username})  #usernmae
                        app.config['messages'].append((question, response))
                        return render_template('index.html', messages=app.config.get('messages', []))
            else:
                
                username = username
                print(username)
                # Load the CSV data specific to the employee
                df = pd.read_csv(csv_file_path)
                employee_data = df[df['Employee Name'] == username]
                if employee_data.empty:
                    response = "No data found for the employee."
                    chain = None
                else:
                    # Save employee-specific data to a temporary file
                    employee_data_file_path = f'data/Employee_data{username}.csv'
                    employee_data.to_csv(employee_data_file_path, index=False)
                    print(f"Saved employee data to {employee_data_file_path}")
                    
                    # Create a new CSV vector store for employee data
                    employee_csv_vector_store = create_csv_vector_store(employee_data_file_path)
                    chain = create_agent(employee_csv_vector_store)
                
                # Determine the source to use
                source = select_source(question)
                print(f"Selected source: {source}")
                if source == 'pdf' and pdf_chain:
                    chain = pdf_chain
                elif source == 'csv' and chain:
                    # Use the newly created employee data chain
                    pass
                    #chain = chain
                else:
                    print("No valid agents available. Skipping question.")
                    response = answer_prompt.invoke({'input': question, 'context': '', 'username': username}) #username
                    app.config['messages'].append((question, response))
                    return render_template('index.html', messages=app.config.get('messages', []))
            if chain:
                start = time.process_time()
                response = chain.invoke({'input': question, 'username': username}) #username
                print(f"Response time: {time.process_time() - start}")
                response = response['answer']
            else:
                response = "No valid agents available."
        app.config['messages'].append((question, response))
    
    return render_template('index.html', messages=app.config.get('messages', []))




@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if not session.get('is_admin'):
        flash('Access denied.')
        return redirect(url_for('index'))
    return render_template('admin.html')


if __name__ == '__main__':
    app.run()