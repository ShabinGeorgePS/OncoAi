#!/usr/bin/env python
"""Test chatbot functionality"""
from app.chatbot.response_engine import ChatbotEngine

engine = ChatbotEngine()
print("✅ ChatbotEngine initialized successfully!")

# Test welcome response
response = engine.get_welcome_response()
print("\n📱 Welcome Response:")
print(response['content'][:200] + "...")

# Test symptoms intent
response = engine.get_response("what are symptoms of oral cancer?")
print("\n🔍 Symptoms Response:")
print(response['content'][:300] + "...")

# Test prevention intent
response = engine.get_response("how can I prevent cancer?")
print("\n🛡️ Prevention Response:")
print(response['content'][:300] + "...")

# Test FAQ search
response = engine.get_response("how accurate is oncoai?")
print("\n📊 Accuracy Response:")
print(response['content'][:300] + "...")

print("\n✅ All tests passed!")
