"""
Streamlit UI components for ONCOAi Chatbot
Handles chatbot display, chat window, and interactions
"""
import streamlit as st
from .response_engine import ChatbotEngine

def initialize_chatbot_session():
    """Initialize chatbot session state"""
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot_engine' not in st.session_state:
        st.session_state.chatbot_engine = ChatbotEngine()

def render_chatbot_button():
    """Render the floating chat button"""
    # This uses custom HTML/CSS for floating button
    button_html = """
    <style>
    .chatbot-button {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg, #1a6b5e, #16a085);
        color: white;
        border: none;
        font-size: 24px;
        cursor: pointer;
        box-shadow: 0 4px 12px rgba(26,107,94,0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        z-index: 999;
    }
    
    .chatbot-button:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 16px rgba(26,107,94,0.6);
    }
    
    .chatbot-badge {
        position: absolute;
        top: -5px;
        right: -5px;
        background: #c0392b;
        color: white;
        border-radius: 50%;
        width: 24px;
        height: 24px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        font-weight: bold;
    }
    </style>
    """
    st.markdown(button_html, unsafe_allow_html=True)

def render_chat_window():
    """Render the chat window interface"""
    initialize_chatbot_session()
    
    # Chat window container
    chat_css = """
    <style>
    .chat-container {
        position: fixed;
        bottom: 110px;
        right: 30px;
        width: 380px;
        max-height: 500px;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.15);
        display: flex;
        flex-direction: column;
        z-index: 998;
        border: 1px solid #e2e8e6;
        font-family: 'DM Sans', sans-serif;
    }
    
    .chat-header {
        background: linear-gradient(135deg, #1a6b5e, #16a085);
        color: white;
        padding: 16px;
        border-radius: 12px 12px 0 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .chat-header h3 {
        margin: 0;
        font-size: 16px;
        font-weight: 600;
    }
    
    .close-btn {
        background: none;
        border: none;
        color: white;
        font-size: 20px;
        cursor: pointer;
        padding: 0;
    }
    
    .chat-messages {
        flex: 1;
        overflow-y: auto;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 12px;
    }
    
    .message {
        padding: 12px 14px;
        border-radius: 8px;
        word-wrap: break-word;
        max-width: 90%;
    }
    
    .message.user {
        background: #e8f5e9;
        margin-left: auto;
        border-radius: 12px 0px 12px 12px;
        color: #2d5016;
    }
    
    .message.bot {
        background: #f5f7f6;
        margin-right: auto;
        border-radius: 0px 12px 12px 12px;
        color: #2c3e50;
        border: 1px solid #e2e8e6;
    }
    
    .chat-input-area {
        padding: 12px;
        border-top: 1px solid #e2e8e6;
        display: flex;
        gap: 8px;
    }
    
    .chat-input {
        flex: 1;
        padding: 10px 12px;
        border: 1px solid #d5dbd7;
        border-radius: 6px;
        font-family: 'DM Sans', sans-serif;
        font-size: 14px;
    }
    
    .send-btn {
        background: #1a6b5e;
        color: white;
        border: none;
        padding: 10px 16px;
        border-radius: 6px;
        cursor: pointer;
        font-weight: 600;
    }
    
    .send-btn:hover {
        background: #0f4d40;
    }
    
    .quick-buttons {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        padding: 0 8px 8px;
    }
    
    .quick-btn {
        padding: 6px 12px;
        background: #f0f3f2;
        border: 1px solid #d5dbd7;
        border-radius: 16px;
        cursor: pointer;
        font-size: 12px;
        color: #1a6b5e;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .quick-btn:hover {
        background: #e8ecea;
        border-color: #1a6b5e;
    }
    </style>
    """
    st.markdown(chat_css, unsafe_allow_html=True)


def display_chat_interface():
    """Display the chat interface with messages"""
    initialize_chatbot_session()
    
    # Create columns for chat window (right side)
    col1, col2 = st.columns([1, 0.4])
    
    with col2:
        # Create a container for chat
        with st.container():
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Header
            st.markdown(
                '<div class="chat-header">'
                '<h3>💬 ONCOAi Assistant</h3>'
                '<button class="close-btn" onclick="window.scrollTo(0, 0)">×</button>'
                '</div>',
                unsafe_allow_html=True
            )
            
            # Messages display
            st.markdown('<div class="chat-messages">', unsafe_allow_html=True)
            
            if not st.session_state.chat_history:
                # Welcome message
                welcome = st.session_state.chatbot_engine.get_welcome_response()
                st.markdown(
                    f'<div class="message bot">{welcome["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                # Display chat history
                for msg in st.session_state.chat_history:
                    msg_class = "user" if msg['role'] == 'user' else 'bot'
                    st.markdown(
                        f'<div class="message {msg_class}">{msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Input area
            st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
            
            user_input = st.text_input(
                "Your message",
                label_visibility="collapsed",
                key="chatbot_input",
                placeholder="Ask me anything..."
            )
            
            if user_input:
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Get bot response
                response = st.session_state.chatbot_engine.get_response(user_input)
                
                # Format response content
                bot_message = response['content']
                if response.get('buttons'):
                    bot_message += "\n\n**Quick options:**"
                
                st.session_state.chat_history.append({
                    'role': 'bot',
                    'content': bot_message
                })
                
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


def render_simple_chat():
    """Render a simplified chat interface inline"""
    initialize_chatbot_session()
    
    st.subheader("💬 ONCOAi Chat Assistant")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    # Input
    user_input = st.chat_input("Ask a question about oral cancer...")
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        # Get response
        response = st.session_state.chatbot_engine.get_response(user_input)
        
        # Add bot message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response['content']
        })
        
        st.rerun()


def render_quick_links():
    """Render quick access buttons"""
    st.markdown("### **Quick Links**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    button_style = """
    <style>
    .quick-link-btn {
        background: linear-gradient(135deg, #1a6b5e, #16a085);
        color: white;
        padding: 16px;
        text-align: center;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s;
    }
    .quick-link-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26,107,94,0.3);
    }
    </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    with col1:
        if st.button("🔍 Symptoms", use_container_width=True):
            st.session_state.chat_history = []
            response = st.session_state.chatbot_engine.get_symptoms_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['content']
            })
            st.rerun()
    
    with col2:
        if st.button("⚠️ Risk Factors", use_container_width=True):
            st.session_state.chat_history = []
            response = st.session_state.chatbot_engine.get_risk_factors_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['content']
            })
            st.rerun()
    
    with col3:
        if st.button("🛡️ Prevention", use_container_width=True):
            st.session_state.chat_history = []
            response = st.session_state.chatbot_engine.get_prevention_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['content']
            })
            st.rerun()
    
    with col4:
        if st.button("📋 When to See Doctor", use_container_width=True):
            st.session_state.chat_history = []
            response = st.session_state.chatbot_engine.get_when_doctor_response()
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response['content']
            })
            st.rerun()
