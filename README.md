# AI Analyst Agent - Multi-Stage Data Analysis Platform

A full-stack application that provides AI-powered data analysis through an intuitive chat interface. The system combines a FastAPI backend with a modern React frontend to deliver real-time, interactive data insights.

## 🎯 Overview

The AI Analyst Agent is an intelligent data analysis platform that allows users to upload datasets and ask questions in natural language. The system automatically generates Python analysis scripts, executes them safely, and provides comprehensive insights with beautiful visualizations and tables.

### ✨ Key Features

- **🤖 Intelligent Code Generation**: AI automatically writes Python analysis scripts based on user questions
- **📊 Beautiful Table Rendering**: Professional HTML tables with dark/light mode support
- **⚡ Real-time Streaming**: Live streaming of AI responses for immediate feedback
- **🔒 Secure Execution**: Safe sandbox environment for running generated code
- **🎨 Modern UI**: Clean, responsive interface with syntax highlighting and collapsible code blocks
- **📈 Multi-format Support**: Handles CSV, Excel (XLS/XLSX) files up to 50MB
- **🌙 Theme Support**: Seamless dark/light mode switching

## 🏗 Architecture

### Backend (FastAPI + Python)
- **Multi-stage Agentic Workflow**: Plan → Generate → Execute → Analyze
- **LLM Integration**: Google Gemini and Together.ai API support
- **WebSocket Streaming**: Real-time response delivery
- **Secure Code Execution**: Containerized Python script execution
- **Data Processing**: Pandas-based analysis with parquet optimization

### Frontend (React + TypeScript)
- **Modern React**: Built with TypeScript and Vite
- **UI Components**: shadcn/ui component library
- **Real-time Communication**: WebSocket-based streaming
- **Syntax Highlighting**: react-syntax-highlighter for code display
- **Responsive Design**: Tailwind CSS with mobile support

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker (for secure code execution)

### Backend Setup

1. **Clone and Navigate**
   ```powershell
   git clone <repository-url>
   cd Agent_Data_Analyst\backend
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Environment Configuration**
   Create `.env` file:
   ```env
   # LLM Provider (choose one)
   GEMINI_API_KEY=your_gemini_key_here
   TOGETHER_API_KEY=your_together_key_here
   
   # Configuration
   LLM_PROVIDER=google  # or "together"
   LLM_TEMPERATURE=0.3  # default 0.3; lower = more deterministic
   GEMINI_MODEL=gemini-2.5-flash
   TOGETHER_MODEL=meta-llama/Llama-3.3-70B-Instruct-Turbo-Free
   ENABLE_LLM=true
   
   # Redis Sessions (optional)
   ENABLE_REDIS_SESSIONS=false  # set to true to enable Redis in prod
   REDIS_URL=redis://localhost:6379/0  # use rediss://<...> on Render managed Redis
   SESSION_TTL_SECONDS=1800  # 30 min default; sliding TTL refreshed on activity
   REDIS_KEY_PREFIX=ai-da
   # Caps to avoid unbounded growth (we don't keep history)
   REDIS_MAX_MESSAGES=10
   REDIS_MAX_ARTIFACTS=10
   ```

4. **Start Backend Server**
   ```powershell
   python backend/run_server.py
   ```

### Local Redis (Dev)

1. **Start Redis with Docker Compose**
   ```powershell
   docker compose up -d redis
   ```
2. **Create `.env` in project root with:**
   ```env
   ENABLE_REDIS_SESSIONS=true
   REDIS_URL=redis://localhost:6379/0
   SESSION_TTL_SECONDS=1800
   REDIS_KEY_PREFIX=ai-da-local
   # LLM controls
   LLM_TEMPERATURE=0.3
   ```
3. **Run the backend** (from project root)
   ```powershell
   python backend/run_server.py
   ```
   The server will log: "Connecting to Redis (URL hidden)... Redis connection successful."

### Frontend Setup

1. **Navigate to Frontend**
   ```powershell
   cd ..\Frontent
   ```

2. **Install Dependencies**
   ```powershell
   npm install
   ```

3. **Start Development Server**
   ```powershell
   npm run dev
   ```

4. **Access Application**
   Open browser to `http://localhost:5173`

## 📋 API Documentation

### Core Endpoints

#### Health Check
```http
GET /health
```
Returns server status and configuration.

#### File Upload
```http
POST /api/upload
Content-Type: multipart/form-data

{
  "file": <binary_file_data>
}
```
**Response:**
```json
{
  "sessionId": "<uuid>",
  "fileId": "<uuid>",
  "status": "analyzed",
  "rows": 1000,
  "columns": ["col1", "col2"]
}
```

#### WebSocket Chat
```ws
WS /api/chat/ws
```
**Message Format:**
```json
{
  "sessionId": "uuid-string",
  "message": "What are the top 5 products by sales?"
}
```

**Response Streaming:**
```json
{"type": "stage", "value": "Planning analysis..."}
{"type": "code", "value": "import pandas as pd\n..."}
{"type": "content", "value": "Based on the analysis..."}
{"type": "done"}
```
Notes:
- `content` chunks may include Markdown (headings, bullets) and HTML tables for structured data.

## 🛠 Technology Stack

### Backend Technologies
- **FastAPI**: Modern Python web framework
- **WebSockets**: Real-time communication
- **Pandas**: Data manipulation and analysis
- **Docker**: Secure code execution sandbox
- **Google Gemini**: Advanced LLM for code generation (gemini-2.5-flash)
- **Together.ai**: Alternative LLM provider (Llama-3.3-70B-Instruct-Turbo-Free)
- **Redis**: Optional persistent session storage
- **Uvicorn**: ASGI server

### Frontend Technologies
- **React 18**: Modern UI library
- **TypeScript**: Type-safe JavaScript
- **Vite**: Fast build tool and dev server
- **Tailwind CSS**: Utility-first CSS framework
- **shadcn/ui**: High-quality component library
- **react-syntax-highlighter**: Code syntax highlighting
- **Lucide React**: Beautiful icon library

## 🔧 Development Features

### Code Generation Workflow
1. **Planning**: AI analyzes the question and dataset schema
2. **Code Generation**: Creates optimized Python analysis script
3. **Safety Validation**: Checks code for security risks
4. **Execution**: Runs code in isolated Docker container
5. **Analysis**: AI interprets results and creates insights

### Advanced Prompting
- **Strict Output Focus**: AI analyzes script execution results first, using dataset schema only for context
- **Period-Aware Analysis**: Explicitly detects year-like period labels (e.g., 2024, 2025) from execution logs and instructs the model to use the exact labels present (no substitutions)
- **Formatted Markdown Output**: Responses follow a clear Markdown structure with sections: `## Key Findings`, `## Results Table` (only if tabular data exists), `## Interpretation`, `## Caveats`, `## Next Steps`
- **Table Generation**: Automatic HTML table creation for structured data, with class `analysis-table` for styling
- **Error Handling**: Graceful fallbacks for analysis failures

### UI/UX Enhancements
- **Collapsible Code Blocks**: Expandable script display
- **Real-time Streaming**: Character-by-character response delivery
- **Theme-Aware Tables**: CSS variable-based styling
- **File Type Detection**: Smart file format handling
- **Responsive Design**: Mobile-friendly interface
- **Markdown Rendering**: Chat UI should render Markdown sections and allow basic HTML tables; ensure `<table class='analysis-table'>` is styled in frontend CSS

## 📊 Sample Usage

1. **Upload Dataset**: Drag and drop CSV/Excel file
2. **Ask Questions**: 
   - "Compare sales between 2024 and 2025"
   - "What are the top 10 products by revenue?"
   - "Show me monthly trends in customer acquisition"
3. **View Results**: Get formatted tables, insights, and generated code
4. **Iterate**: Ask follow-up questions for deeper analysis

## 🔒 Security Considerations

- **Sandboxed Execution**: All user code runs in isolated Docker containers
- **Input Validation**: File type and size restrictions
- **HTML Sanitization**: Safe rendering of LLM-generated content
- **No Network Access**: Analysis scripts cannot make external calls
- **Temporary Storage**: Session data automatically cleaned up

## 🚧 Development Roadmap

### Completed ✅
- Multi-stage agentic workflow
- Real-time WebSocket streaming
- HTML table generation and rendering
- Collapsible code blocks
- Dark/light theme support
- Secure code execution
- File upload and processing
- Redis-backed persistent sessions
- Graceful Redis connection handling

## ☁️ Deployment Notes (Render backend + Vercel frontend)

- **Redis configuration (Render)**
  - Provision a managed Redis instance on Render.
  - Set env vars on the backend service:
    - `ENABLE_REDIS_SESSIONS=true`
    - `REDIS_URL=rediss://<from Render>` (prefer TLS)
    - `SESSION_TTL_SECONDS=1800` (or your choice)
    - `REDIS_KEY_PREFIX=ai-da`
    - `REDIS_MAX_MESSAGES=10`
    - `REDIS_MAX_ARTIFACTS=10`
  - Keep a single instance (no autoscaling/scale-to-zero) so uploaded dataset files remain accessible during the session.

- **App behavior**
  - Sessions are created via `POST /api/upload` and last until the user refreshes the page.
  - The frontend must pass the returned `sessionId` with every REST/WS call.
  - WebSocket model: per-turn connection (open for a prompt, close when done). Persistent WS is optional and not required.

- **Security/logging**
  - The backend avoids logging full Redis URLs (credentials are masked).

## ✅ Session Reliability Summary

- Redis is initialized on app startup (`backend/app/main.py`).
- Sliding TTL keeps sessions alive while active; lists are trimmed to last N.
- Use `rediss://` on Render and keep a single instance to avoid file-path issues across machines.
- Frontend uses the real `sessionId` returned by `/api/upload` for all REST and WS calls (no fake IDs).
- Backend validates `sessionId` as a UUID and returns 400 for invalid formats.

### In Progress 🔄
- Enhanced error handling
- Performance optimizations
- Additional chart types

### Planned 📋
- **Export Capabilities**: Download results as PDF/Excel
- **Chart Integration**: Matplotlib/Plotly visualizations
- **Multi-file Support**: Analyze multiple datasets
- **Collaboration Features**: Share analysis sessions
- **Advanced Analytics**: Statistical testing, ML models
- **Database Integration**: Connect to external data sources
