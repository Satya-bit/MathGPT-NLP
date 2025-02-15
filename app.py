import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

##Set up the streamlit app
st.set_page_config(page_title="Text To Math Problem Solver And Data Search Assistant")

# st.title("Text To Math Problem Solver Using Google Gemma")
st.markdown("<h1 style='text-align: center; color: black;'>Text To Math Problem Solver Using Google Gemma</h1>", unsafe_allow_html=True)

groq_api_key=st.sidebar.text_input(label="GROQ API Key",type="password")

if not groq_api_key:
    st.info("Please add your GROQ API key")
    st.stop() #If API key is not provided then stop the app. The code will not execute further
    
llm=ChatGroq(model="gemma2-9b-It",groq_api_key=groq_api_key)

##Initialize thje tool
#Just for seraching like formulas we will using wikipedia tools
wikipedia_wrapper=WikipediaAPIWrapper() #To access the wikipedia api
wikipedia=Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="A tool for searching the Internet to find the various information on the topics mentioned"
    
)

## Initialize the math tool
math_chain=LLMMathChain.from_llm(llm=llm) #This math chain is used for calculation and it will be interact with our llm
calculator=Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related questions. Only input mathematical expression need to be provided."
)

prompt="""
You are a agent tasked for solving users mathematical questions. Logically arrive at the the solution and 
provide a detailed explanation and diplay it pointwise 
for the question below
Question:{question}
Answer:
"""

prompt_template=PromptTemplate(
    input_variables=["question"],
    template=prompt
)


chain=LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool=Tool(
    name="Reasoning tool",
    func=chain.run,
    description="A tool for answering logic based and reasoning questions"
) #Converted chain into tool. This is like custom tool besides the tools provided by langchain which are wiki and math. Refer tools section.

##Initialize the agents

assistant_agent=initialize_agent(
    tools=[wikipedia,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, #A zero shot agent that does a reasoning step before acting.It does not guess blindly
    verbose=False,
    handle_parsing_errors=True#Without handle_parsing_errors → AI might crash if it gives an invalid response. With handle_parsing_errors → AI can catch errors and try again or give a better response.
)
if "messages" not in st.session_state: #If there are no messages in the session then display the below content
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I'm a math chatbot who can answer all your maths questions"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"]) #For displaying the chat history
    
#Function to generate the response

# def genearte_response(question):
#     response=assistant_agent.invoke({'input':question})#invoke method is used to call a chain, model, or agent and get a response
#     return response

# LETS start the interaction
question=st.text_area("Enter your question:","I have 10 apples and 5 oranges. I gave 2 apples and 1 orange.How many fruits do I have in total now?") #This is the input box for the user to enter the question

if st.button("find my answer"):
    if question:
        with st.spinner("Generating response..."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            
            st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response=assistant_agent.run(question,callbacks=[st_cb])
            
            st.session_state.messages.append({"role":"assistant","content":response})
            st.write('### Response:')
            st.success(response)
    else:
        st.info("Please enter a question")
            
            
    