# 🎨 AccessiBot
### An AI-Powered Adobe Express Add-On for Inclusive Design

## 🔗 [Devpost](https://devpost.com/software/accessibot)

## 🏆 Awards
- **[MLH] Most Creative Adobe Express Add-On** - HackCOMS @ RIT - Nov 2024

## 🌟 Overview
AccessiBot is an Adobe Express Add-On that revolutionizes the way creators approach accessible design. With just a single click, our tool analyzes your design and provides actionable insights to make your content more accessible to everyone.

## ✨ Features
- 🔍 **Font Analysis**: Evaluates clarity and visibility of text elements
- 🖼️ **Image Assessment**: Checks image visibility and contrast
- 🎨 **Color Contrast**: Ensures WCAG compliance
- 📖 **Text Readability**: Analyzes and suggests improvements
- 🔤 **Font Style Optimization**: Recommends better font choices
- ♿ **Accessibility Enhancements**: Provides content modification suggestions

## 🛠️ Technology Stack

### Backend Server
- FastAPI for high-performance API endpoints
- LangChain for AI workflow orchestration
- TogetherAI integration

### AI Models
- LLama 3.2 90B Vision Instruct
- LLama 3.2 11B Vision
- Qwen 2.5 72B Instruct

### Frontend
- Adobe Express Add-On SDK
- JavaScript

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Adobe Express Developer Account
- TogetherAI API key
- ngrok authtoken

### Backend Setup
1. Clone the repository
```bash
git clone https://github.com/AakashKhatu/AccesiBot-HackCOMS
cd AccesiBot-HackCOMS/API
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Set up environment variables
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

4. Run the server
```bash
python fastApiServer.py
```

### Adobe Express Add-On Setup
1. Navigate to the add-on directory
```bash
cd frontend
```

2. Install dependencies
```bash
npm install
```

3. Configure the add-on
- Open Adobe Express Add-on Developer Console
- Create a new project
- Upload the add-on files
- Enable debug mode for testing

## 💡 How It Works
1. User clicks the "Analyze" button in Adobe Express
2. Add-on captures the current page content
3. Content is processed by our AI model pipeline:
   - Vision analysis for image and layout assessment
   - Text analysis for readability and accessibility
   - Color analysis for contrast and visibility
4. Results are presented with actionable improvements

## 👥 Meet the Team

<div align="center">

| ![Aakash Khatu](https://avatars.githubusercontent.com/u/25435412?v=4) | ![Gayatri Khandagale](https://avatars.githubusercontent.com/u/47130021?v=4) |
|:----------------------------------------:|:----------------------------------------:|
| [Aakash Khatu](https://github.com/AakashKhatu) | [Gayatri Khandagale](https://github.com/Gayatri-K) |
| AI & API Development | AI & Add on Development |

</div>


## 🙏 Acknowledgments
- Adobe Express team for their excellent developer tools
- The open-source AI community & Meta Llama for their absolutely stunning multimodal LLM Models
- Major League Hacking (MLH) for recognizing our innovation!

---
<div align="center">
Made with ❤️ for accessible design

[Report Bug](https://github.com/AakashKhatu/AccesiBot-HackCOMS/issues) · [Request Feature](https://github.com/AakashKhatu/AccesiBot-HackCOMS/issues)
</div>
