HR Chatbot Using Flask and LangChain: A Professional Web Application
The HR Chatbot is a robust web application designed to streamline HR-related queries by leveraging cutting-edge AI technology. Built on Flask, the chatbot integrates LangChain to deliver dynamic and accurate responses, offering a seamless user experience for employees and administrators.

Key Features:
Natural Language Processing (NLP):

Powered by Groq's ChatGroq and Google's Generative AI, the chatbot uses advanced NLP for understanding and answering complex HR-related queries.
Provides responses based on structured and unstructured data, including company policies and HR documents.
PDF Integration:

The chatbot can retrieve answers from a comprehensive PDF of company policies, making it easy for users to get information without browsing through lengthy documents manually.
Users can simply ask questions like "What is the company's leave policy?" and get accurate responses from the document.
MongoDB Integration:

All employee-related data, including attendance and leave information, is stored in MongoDB.
Employees can inquire about personal details such as remaining annual leave, casual leave, and other HR-related data.
Role-based Access:

Admins: Can access all employee data, including attendance records, leave balances, and other sensitive information.
Employees: Limited access to personal data only, allowing them to view their attendance, remaining leaves, and retrieve policy information from PDFs.
Email Functionality:

Employees and administrators can send emails directly from the chatbot.
Simply provide the content, and the chatbot will craft and send a professional email to the recipient, improving communication efficiency within the organization.
AI-Powered Vector Embeddings:

The application utilizes vector embeddings for document-based queries. This ensures precise and contextually relevant answers by creating vector representations of PDF content and MongoDB-stored data.
Efficient and User-friendly:

Designed to cater to both tech-savvy and non-technical users, the chatbot offers a smooth, conversational interface. It is responsive, fast, and intuitive, making HR processes easier for both employees and HR personnel.
Example Scenarios:
Employee: “How many annual leave days do I have left?”

The bot will fetch data from MongoDB and provide the exact number of remaining days.
Admin: “Show me John Doe's attendance records for this month.”

The bot retrieves and displays John's attendance records, ensuring the admin has full access to critical data.
Email Assistance: “Send an email to HR requesting approval for leave extension.”

The chatbot will craft a professional email based on the user's input and send it to the appropriate HR contact.
This professional HR Chatbot not only enhances productivity but also improves employee satisfaction by simplifying HR-related queries and tasks.
