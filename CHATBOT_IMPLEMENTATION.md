# ONCOAi Chatbot - Implementation Guide

## ✅ What Was Built

Your ONCOAi application now includes a **rule-based AI chatbot assistant** that helps users understand oral cancer, get guidance on detection results, and access preventive health information.

---

## 📦 Project Structure

```
app/
├── chatbot/
│   ├── __init__.py                 # Module entry point
│   ├── response_engine.py          # Core chatbot logic (rule-based)
│   ├── ui_components.py            # Streamlit UI elements
│   └── data/
│       ├── symptoms.json           # Oral cancer symptoms database
│       ├── faq.json               # Frequently asked questions
│       └── guidance.json           # Post-detection guidance workflows
```

---

## 🤖 Chatbot Features

### 1. **Oral Cancer Symptoms** (6 Categories)
- Persistent Oral Ulcer
- White or Red Patches
- Difficulty Swallowing
- Persistent Mouth Pain
- Swelling or Lump
- Jaw Problems

Each includes:
- Detailed description
- Warning signs to watch for
- Prevention tips

### 2. **Risk Factors** (8 Categories)
- Tobacco Use (HIGH RISK)
- Alcohol Consumption (HIGH RISK)
- Age
- HPV Infection (HIGH RISK)
- Poor Oral Hygiene
- Sun Exposure
- Previous Cancer History
- Weakened Immune System

### 3. **Prevention Guidance**
Complete prevention checklist:
- Quit tobacco
- Limit alcohol
- Maintain oral hygiene
- Use SPF lip balm
- Nutrition guidance
- Regular checkups
- HPV vaccination info

### 4. **FAQs** (17 Questions Across 4 Categories)

**About Oral Cancer:**
- What is oral cancer?
- How common is it?
- Early detection curability
- Survival rates

**Using ONCOAi:**
- Accuracy of AI model
- Confidence score explanation
- False positive/negative possibility
- Image quality requirements
- Data privacy & security

**After Detection:**
- What to do if CANCER detected
- What if SUSPICIOUS result
- Biopsy process
- Treatment options

**Prevention & Health:**
- Prevention methods
- HPV vaccine benefits
- Checkup frequency
- Self-examination guide

### 5. **Post-Detection Guidance**

**For CANCER Results:**
- 6-step action plan
- Follow-up timelines
- Red flags requiring emergency care

**For SUSPICIOUS Results:**
- Don't worry reassurance
- 5-step guidance
- Possible causes explained
- Investigation timeline

**For NON-CANCER Results:**
- Positive confirmation
- Next steps if lesion persists
- Prevention recommendations
- Ongoing monitoring guidance

---

## 🎯 How It Works

### Rule-Based Intent Detection

The chatbot uses **keyword matching** to understand user intent:

```
User Input → Intent Detection → Knowledge Base Lookup → Response Generation
```

**Intent Categories:**
- `symptoms` - User asking about symptoms
- `risk_factors` - Risk factor questions
- `when_doctor` - When to see a doctor
- `prevention` - Prevention methods
- `confidence_score` - Understanding confidence scores
- `after_cancer` - After positive results
- `accuracy` - Model accuracy questions
- `faq` - General FAQ search

### Example Conversations

**User:** "What are oral cancer symptoms?"
**Bot:** Returns 6 symptoms with details, warning signs, and prevention tips

**User:** "How accurate is ONCOAi?"
**Bot:** Explains confidence scores with interpretation ranges

**User:** "What should I do if cancer is detected?"
**Bot:** 6-step action plan with emergency red flags

---

## 📱 UI Components

### Quick Access Buttons
Located in main app interface:
- 🔍 Symptoms
- ⚠️ Risk Factors
- 🛡️ Prevention
- 📋 When to See Doctor

### Chat Interface
- Expandable chat window
- Message history display
- Real-time response generation
- Clean, professional styling

### Integration Points
```python
# In app.py
from chatbot import initialize_chatbot_session, render_simple_chat, render_quick_links

# Initialize session
initialize_chatbot_session()

# Display quick links
render_quick_links()

# Display full chat interface
render_simple_chat()
```

---

## 📊 Knowledge Base Contents

### Symptoms Database (`symptoms.json`)
- 6 major symptoms
- 8 risk factors with severity levels
- 9 "when to see doctor" indicators

### FAQ Database (`faq.json`)
- 17 Q&A pairs
- 4 main categories
- Comprehensive answers

### Guidance Database (`guidance.json`)
- 3 diagnosis workflows (CANCER, SUSPICIOUS, NON-CANCER)
- Step-by-step instructions
- Daily/Weekly/Monthly/Annual health steps

---

## 🚀 How to Use

### For End Users

1. **On the main app page**, scroll to "Need Help? Ask Our AI Assistant"
2. **Click any quick button** (Symptoms, Risk Factors, etc.) for instant responses
3. **Or open the "Chat with ONCOAi Assistant"** expander
4. **Type your question** and get instant AI-powered answers

### For Developers

```python
# Import and initialize
from chatbot import ChatbotEngine

engine = ChatbotEngine()

# Get response
response = engine.get_response("what are symptoms?")
# Returns: { 'type': 'text', 'content': '...', 'buttons': [...] }

# Or get specific responses
response = engine.get_prevention_response()
response = engine.get_risk_factors_response()
response = engine.get_symptoms_response()
```

---

## ✨ Key Advantages

✅ **No API Calls Required** - Fully self-contained, rule-based system
✅ **Fast Responses** - Instant answers from local knowledge base
✅ **Evidence-Based** - All information from medical sources
✅ **User-Friendly** - Simple language, emoji indicators, structured answers
✅ **Comprehensive** - Covers symptoms, prevention, guidance, and FAQs
✅ **Scalable** - Easy to add new symptoms, FAQs, or guidance workflows
✅ **Privacy-First** - No data collection or external API calls
✅ **Integrated** - Seamlessly embedded in your Streamlit app

---

## 🔧 Customization

### Add New FAQ

Edit `app/chatbot/data/faq.json`:
```json
{
  "category": "Your Category",
  "questions": [
    {
      "id": "unique_id",
      "q": "Your question?",
      "a": "Your answer..."
    }
  ]
}
```

### Add New Symptom

Edit `app/chatbot/data/symptoms.json`:
```json
{
  "id": "symptom_id",
  "name": "Symptom Name",
  "description": "Description...",
  "warning_signs": ["Sign 1", "Sign 2"],
  "prevention": "Prevention tips..."
}
```

### Add New Intent

Edit `response_engine.py`:
```python
# Add to intent_patterns
self.intent_patterns['new_intent'] = ['keyword1', 'keyword2']

# Add response method
def get_new_intent_response(self):
    return {'type': 'text', 'content': '...'}

# Add to response_map
response_map['new_intent'] = self.get_new_intent_response
```

---

## 📈 Response Examples

### Symptom Query
**Input:** "what are symptoms?"
**Output:** All 6 symptoms with detailed descriptions

### Risk Factor Query
**Input:** "risk factors"
**Output:** All 8 risk factors with severity levels

### Prevention Query
**Input:** "how to prevent"
**Output:** 8-step prevention checklist

### After Detection Query
**Input:** "what if cancer"
**Output:** 6-step action plan with timelines

---

## 🧪 Testing

Run the test script to verify functionality:
```bash
python test_chatbot.py
```

**Tests Included:**
- ✅ ChatbotEngine initialization
- ✅ Welcome response
- ✅ Symptoms intent detection
- ✅ Prevention intent detection
- ✅ FAQ search
- ✅ Confidence score responses
- ✅ All response formatting

---

## 📝 Technical Details

- **Framework:** Streamlit (Python)
- **Type:** Rule-based chatbot (no ML, no APIs)
- **Data Storage:** JSON files (local, no database needed)
- **Intent Detection:** Keyword matching with pattern library
- **Response Generation:** Template-based with dynamic content

---

## 🎨 UI Styling

All UI components are styled with:
- Professional teal color scheme (#1a6b5e)
- Responsive design
- Clear typography (DM Sans font)
- Emoji indicators for visual clarity
- Proper spacing and alignment

---

## 🔐 Privacy & Security

✅ **No external APIs** - All processing local
✅ **No data collection** - Chat history cleared on page refresh
✅ **No authentication needed** - Same auth as main app
✅ **No tracking** - Complete user privacy

---

## 📞 Support & Maintenance

The chatbot system is designed to be:
- **Easy to maintain** - JSON-based knowledge base
- **Easy to update** - No code changes needed for content updates
- **Easy to extend** - Modular architecture allows easy additions
- **Performant** - Rule-based approach with O(n) complexity

---

## 🎓 Learning Resources

The chatbot includes comprehensive information about:
- Oral cancer detection and prevention
- Understanding AI prediction confidence
- When to seek medical help
- Medical guidance after diagnostic results
- Lifestyle factors and risk reduction

All information is sourced from medical standards and best practices.

---

**Implementation Date:** March 27, 2026  
**Version:** 1.0  
**Status:** ✅ Fully Functional and Tested
