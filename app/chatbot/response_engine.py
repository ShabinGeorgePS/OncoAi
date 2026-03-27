"""
Rule-based chatbot response engine for ONCOAi
Handles intent detection and response generation
"""
import json
import os
from typing import Dict, List, Tuple

class ChatbotEngine:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        self.symptoms = self._load_json('symptoms.json')
        self.faqs = self._load_json('faq.json')
        self.guidance = self._load_json('guidance.json')
        
        # Intent patterns for rule matching
        self.intent_patterns = {
            'symptoms': [
                'what are symptoms', 'signs of', 'oral cancer symptoms', 'how to detect',
                'what to look for', 'warning signs', 'early signs', 'symptoms of cancer'
            ],
            'risk_factors': [
                'risk factors', 'what causes', 'who is at risk', 'can i get cancer',
                'risk of oral cancer', 'likely to get'
            ],
            'when_doctor': [
                'when to see doctor', 'should i see doctor', 'when should i go',
                'need appointment', 'when to contact doctor', 'doctor appointment'
            ],
            'prevention': [
                'prevent cancer', 'how to prevent', 'prevention tips', 'stay healthy',
                'avoid cancer', 'lower risk', 'preventive measures'
            ],
            'confidence_score': [
                'what is confidence', 'confidence score', 'what does % mean',
                'how accurate', 'reliability', 'trust the result'
            ],
            'after_cancer': [
                'what to do if cancer', 'positive result', 'what if cancer detected',
                'next steps after', 'treatment options', 'after detection'
            ],
            'after_suspicious': [
                'what if suspicious', 'suspicious result', 'what does suspicious mean',
                'should i worry', 'after suspicious'
            ],
            'accuracy': [
                'how accurate is', 'reliability', 'false positive', 'false negative',
                'can it be wrong', 'mistakes', 'errors'
            ],
            'faq': [
                'faq', 'frequently asked', 'common questions', 'help', 'information',
                'tell me about', 'explain', 'what is', 'how does'
            ]
        }
    
    def _load_json(self, filename: str) -> Dict:
        """Load JSON data file"""
        try:
            with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
            return {}
    
    def detect_intent(self, user_input: str) -> str:
        """Detect user intent from input"""
        user_input_lower = user_input.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in user_input_lower:
                    return intent
        
        return 'faq'  # Default to FAQ
    
    def get_symptoms_response(self) -> Dict:
        """Generate response about symptoms"""
        symptoms = self.symptoms.get('symptoms', [])
        
        symptoms_text = "🔍 **Common Oral Cancer Symptoms:**\n\n"
        for sym in symptoms:
            symptoms_text += f"• **{sym['name']}** - {sym['description']}\n"
        
        return {
            'type': 'text',
            'content': symptoms_text,
            'buttons': [
                {'label': '⚠️ Risk Factors', 'intent': 'risk_factors'},
                {'label': '📋 When to See Doctor', 'intent': 'when_doctor'},
                {'label': '🛡️ Prevention', 'intent': 'prevention'}
            ]
        }
    
    def get_risk_factors_response(self) -> Dict:
        """Generate response about risk factors"""
        risk_factors = self.symptoms.get('risk_factors', [])
        
        risk_text = "⚠️ **Risk Factors for Oral Cancer:**\n\n"
        for rf in risk_factors:
            risk_level = rf.get('risk_level', '')
            risk_text += f"• **{rf['category']}** [{risk_level}]\n  {rf['details']}\n"
        
        return {
            'type': 'text',
            'content': risk_text,
            'buttons': [
                {'label': '🛡️ Prevention Tips', 'intent': 'prevention'},
                {'label': '📋 Symptoms', 'intent': 'symptoms'}
            ]
        }
    
    def get_prevention_response(self) -> Dict:
        """Generate response about prevention"""
        prevention_text = "🛡️ **How to Prevent Oral Cancer:**\n\n"
        prevention_text += "✓ Quit tobacco (smoking, chewing)\n"
        prevention_text += "✓ Limit alcohol consumption\n"
        prevention_text += "✓ Maintain excellent oral hygiene\n"
        prevention_text += "✓ Use SPF lip balm (especially outdoors)\n"
        prevention_text += "✓ Eat fruits and vegetables (rich in antioxidants)\n"
        prevention_text += "✓ Get regular dental checkups (every 6 months)\n"
        prevention_text += "✓ Consider HPV vaccination if not vaccinated\n"
        prevention_text += "✓ Practice monthly self-examination\n"
        
        return {
            'type': 'text',
            'content': prevention_text,
            'buttons': [
                {'label': '📋 Self-Examination Guide', 'intent': 'self_exam'},
                {'label': '💉 HPV Vaccine Info', 'intent': 'hpv_vaccine'}
            ]
        }
    
    def get_when_doctor_response(self) -> Dict:
        """Generate response about when to see doctor"""
        when_see = self.symptoms.get('when_to_see_doctor', [])
        
        text = "📋 **When to See a Doctor or Dentist:**\n\n"
        for i, item in enumerate(when_see, 1):
            text += f"{i}. {item}\n"
        
        text += "\n⏱️ **Timeline:** Schedule appointment within 1-2 weeks of noticing symptoms."
        
        return {
            'type': 'text',
            'content': text,
            'buttons': [
                {'label': '🔍 Symptoms', 'intent': 'symptoms'},
                {'label': '🛡️ Prevention', 'intent': 'prevention'}
            ]
        }
    
    def get_confidence_response(self) -> Dict:
        """Generate response about confidence scores"""
        text = "📊 **Understanding Confidence Scores:**\n\n"
        text += "The confidence score (0-100%) shows how certain the AI is about its prediction.\n\n"
        text += "**Score Interpretation:**\n"
        text += "• **85-100%** - High confidence (strong indicator)\n"
        text += "• **60-84%** - Moderate confidence (worth investigating)\n"
        text += "• **<60%** - Lower confidence (less certain)\n\n"
        text += "⚠️ **Important:** Even with high confidence, a doctor's evaluation is required. "
        text += "ONCOAi is a screening tool, not a diagnosis."
        
        return {
            'type': 'text',
            'content': text,
            'buttons': [
                {'label': '❓ After Detection', 'intent': 'faq'},
                {'label': '🏥 What to Do Next', 'intent': 'after_cancer'}
            ]
        }
    
    def get_after_cancer_response(self) -> Dict:
        """Generate response for positive cancer results"""
        guidance = self.guidance.get('guidance_workflows', {}).get('cancer_detected', {})
        
        text = guidance.get('title', '') + "\n\n"
        text += "**Steps to Take:**\n"
        
        for step_data in guidance.get('steps', [])[:5]:  # First 5 steps
            text += f"\n**{step_data['step']}. {step_data['title']}**\n"
            text += f"   {step_data['description']}\n"
            text += f"   → {step_data['action']}\n"
        
        text += "\n🚨 **Seek Emergency Care If:**\n"
        for flag in guidance.get('red_flags', [])[:3]:
            text += f"• {flag}\n"
        
        return {
            'type': 'text',
            'content': text,
            'buttons': [
                {'label': '📞 Doctor Appointment', 'intent': 'faq'},
                {'label': '❓ FAQs', 'intent': 'faq'}
            ]
        }
    
    def search_faq(self, query: str) -> Dict:
        """Search FAQs for matching question"""
        query_lower = query.lower()
        all_faqs = self.faqs.get('faqs', [])
        
        # Try exact match first
        for category in all_faqs:
            for item in category.get('questions', []):
                if query_lower in item['q'].lower() or query_lower in item['a'].lower():
                    text = f"**Q: {item['q']}**\n\n{item['a']}"
                    return {
                        'type': 'text',
                        'content': text,
                        'buttons': [
                            {'label': '📋 More FAQs', 'intent': 'faq'},
                            {'label': '🔍 Symptoms', 'intent': 'symptoms'}
                        ]
                    }
        
        # If no match, return general FAQ intro
        text = "**Frequently Asked Questions:**\n\n"
        text += "I can answer questions about:\n"
        text += "• Oral cancer information\n"
        text += "• Using ONCOAi\n"
        text += "• After detection\n"
        text += "• Prevention & health\n\n"
        text += "What would you like to know?"
        
        return {
            'type': 'text',
            'content': text,
            'buttons': [
                {'label': '🔍 Symptoms', 'intent': 'symptoms'},
                {'label': '⚠️ Risk Factors', 'intent': 'risk_factors'},
                {'label': '🛡️ Prevention', 'intent': 'prevention'},
                {'label': '🏥 After Detection', 'intent': 'after_cancer'}
            ]
        }
    
    def get_welcome_response(self) -> Dict:
        """Generate welcome message"""
        text = "👋 **Welcome to ONCOAi Chat Assistant!**\n\n"
        text += "I'm here to help you understand:\n\n"
        text += "🔍 **Oral Cancer Symptoms** - What to look for\n"
        text += "⚠️ **Risk Factors** - Who's at risk\n"
        text += "🛡️ **Prevention** - How to stay healthy\n"
        text += "📋 **When to See a Doctor** - Timeline & symptoms\n"
        text += "🏥 **After Detection** - What to do next\n"
        text += "❓ **FAQs** - Common questions answered\n\n"
        text += "How can I help you today?"
        
        return {
            'type': 'text',
            'content': text,
            'buttons': [
                {'label': '🔍 Symptoms', 'intent': 'symptoms'},
                {'label': '⚠️ Risk Factors', 'intent': 'risk_factors'},
                {'label': '🛡️ Prevention', 'intent': 'prevention'},
                {'label': '📋 When to See Doctor', 'intent': 'when_doctor'}
            ]
        }
    
    def get_response(self, user_input: str) -> Dict:
        """Get chatbot response based on user input"""
        if not user_input.strip():
            return self.get_welcome_response()
        
        intent = self.detect_intent(user_input)
        
        response_map = {
            'symptoms': self.get_symptoms_response,
            'risk_factors': self.get_risk_factors_response,
            'when_doctor': self.get_when_doctor_response,
            'prevention': self.get_prevention_response,
            'confidence_score': self.get_confidence_response,
            'after_cancer': self.get_after_cancer_response,
            'accuracy': lambda: self.search_faq('how accurate'),
            'faq': lambda: self.search_faq(user_input)
        }
        
        response_func = response_map.get(intent, lambda: self.search_faq(user_input))
        return response_func()
