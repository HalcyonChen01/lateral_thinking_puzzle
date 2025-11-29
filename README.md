# 🐢 Turtle Soup Mystery

An AI-powered lateral thinking puzzle game where players uncover the hidden truth behind biza## 📄 License

This project is for educational and entertainment purposes only.

---

🎮 **Enjoy the game! Remember, the truth is often stranger than you imagine...**

<span style="color: red;">"Remember to answer the questions it gives you carefully — two opposing replies may lead to very different outcomes."</span>ies by asking yes/no questions.

![Game Preview](PIC/Crow.jpg)

## 🎮 About the Game

**Turtle Soup** (also known as Lateral Thinking Puzzles) is a deductive reasoning game. The game presents a seemingly absurd or illogical scenario (the surface), and players must ask questions to deduce the hidden truth (the solution).

### Game Rules
- Players can only ask **yes/no questions**
- The AI will only respond with: **Yes**, **No**, **Irrelevant**, or **Partially correct**
- Piece together the complete story through progressive questioning

## ✨ Features

- 🧩 **10 Carefully Crafted Puzzles** - Progressive difficulty from easy to challenging
- 🤖 **AI-Powered** - Intelligent conversations using DeepSeek API
- 🎨 **Retro Pixel Style** - Nostalgic gaming experience with VT323 font
- 🖼️ **Progressive Image Reveal** - Unlock more of the image with each puzzle solved
- 🔊 **Sound Effects** - Immersive audio experience
- 💡 **Hint System** - Request AI hints when you're stuck
- 📱 **Responsive Design** - Works on desktop and mobile devices
- ⚠️ **Content Warning** - Disclaimer before game start

## 🛠️ Tech Stack

- **Backend**: FastAPI (Python)
- **Frontend**: HTML5 + CSS3 + JavaScript
- **AI**: DeepSeek API (OpenAI SDK compatible)
- **Font**: VT323 (Google Fonts)

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd YN
```

### 2. Create Virtual Environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install fastapi uvicorn python-dotenv openai
```

### 4. Configure Environment Variables
Create a `.env` file:
```env
DEEPSEEK_API_KEY=your_deepseek_api_key
```

> Get your API Key: https://platform.deepseek.com/

### 5. Run the Game
```bash
python main.py
```

Visit http://localhost:8000 to start playing!

## 🌐 Share with Friends

### Option 1: Local Network Sharing
Friends on the same network can access via your IP address:
```
http://your-ip-address:8000
```

### Option 2: Using Ngrok (Public Access)
1. Download [Ngrok](https://ngrok.com/download)
2. Sign up and get your authtoken
3. Run:
```bash
ngrok http 8000
```
4. Share the generated public URL

## 📁 Project Structure

```
YN/
├── main.py              # FastAPI backend server
├── .env                 # Environment variables (API Key)
├── README.md            # Project documentation
├── static/
│   ├── index.html       # Main game page
│   ├── styles.css       # Stylesheet
│   ├── app.js           # Frontend logic
│   └── sounds/          # Sound effects
│       ├── enter.mp3    # Entry sound
│       ├── Success.mp3  # Success sound
│       └── camera.wav   # Special effect sound
└── PIC/
    └── Crow.jpg         # Game image
```

## ⚠️ Important Notes

- This game contains **horror elements** and **jump scares**
- Recommended for players aged 16 and above
- Players with heart conditions should exercise caution
- Camera permission required (for special effects)

## 📝 Developer Notes

- Game uses `host="0.0.0.0"` configuration to support LAN access
- API Key is stored in `.env` file and is not exposed to players
- Hot-reload development mode supported

## 📄 License

This project is for educational and entertainment purposes only.

---

🎮 **Enjoy the game! Remember, the truth is often stranger than you imagine...**

“Remember to answer the questions it gives you carefully — two opposing replies may lead to very different outcomes.”
