# AI Data Analyst 🤖📊

A comprehensive AI-powered data analysis platform that combines secure Python code execution, intelligent workflow orchestration, and modern web interfaces to provide automated data insights and visualizations.

## 🚀 Features

- **🔒 Secure Code Execution**: Advanced AST-based code validation and sandboxed execution environment
- **📊 Intelligent Analysis**: Multi-LLM powered data analysis using Gemini, OpenAI, and Together.ai
- **🔄 Workflow Orchestration**: LangGraph-based workflow management for complex analysis pipelines
- **🌐 REST API**: Comprehensive FastAPI backend with automatic documentation
- **📱 Modern Frontend**: React-based interface with TypeScript and Tailwind CSS
- **⚡ Production Ready**: Docker containerization with comprehensive security hardening
- **🔧 Session Management**: Persistent analysis sessions with comprehensive state tracking

## 🏗️ Architecture

The system consists of a layered architecture with secure execution at its core:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │ Security Layer  │
│   (Port 5173)    │◄──►│   (Port 8000)    │◄──►│ Code Validation │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │                        │
                       ┌────────▼────────┐    ┌──────────▼──────────┐
                       │ Session Storage │    │  Sandbox Execution  │
                       │  In-Memory      │    │  Resource Limited   │
                       └─────────────────┘    └─────────────────────┘
```

## 🛠️ Current Implementation Status

### ✅ Completed Components

#### Security Layer (100% Complete)
- **Code Validator** (`backend/security/code_validator.py`): AST-based validation with 3 security levels
- **Sandbox Environment** (`backend/security/sandbox.py`): Resource-limited execution with platform compatibility
- **Comprehensive Testing**: Validated against malicious code patterns

#### API Backend (100% Complete)  
- **FastAPI Server** (`backend/main_simple.py`): Production-ready with CORS and error handling
- **Session Management**: In-memory session tracking with file upload support
- **File Processing**: Multi-format support (CSV, Excel, JSON, Parquet)
- **Live Deployment**: Running at http://localhost:8000 with Swagger docs

#### Data Processing (100% Complete)
- **Enhanced Data Cleaner**: Advanced preprocessing with type inference
- **Data Profiler**: Comprehensive statistical analysis
- **Type Inference**: Intelligent column type detection

### 🚧 In Progress
- **LangGraph Integration**: Workflow orchestration (planned)
- **Frontend Enhancement**: React interface improvements

## 🛠️ Tech Stack

### Backend
- **FastAPI**: High-performance async API framework
- **LangGraph**: Agent workflow orchestration
- **Redis**: Session state and caching
- **Pandas/NumPy**: Data processing and analysis
- **Docker**: Containerized deployment

### Frontend  
- **React 18**: Modern UI framework
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Vite**: Fast development and build tooling

### LLM Integration
- **Primary**: Google Gemini 2.0 Flash
- **Secondary**: OpenAI GPT-4o-mini  
- **Tertiary**: Anthropic Claude 3.5 Haiku

## 🚀 Quick Start

### Prerequisites
- Python 3.11+ 
- Node.js 18+
- Docker (optional)

### 1. Backend Setup (Ready to Use)

The backend is already configured and ready to run:

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# The server is ready to start
python main_simple.py
```

**🟢 Server Status**: The API will be live at:
- **API**: http://localhost:8000  
- **Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at http://localhost:5173

### 3. Docker Setup (Alternative)
```bash
docker-compose up --build
```

## 📁 Project Structure

```
ai-data-analyst/
├── backend/                    # Python FastAPI backend (READY)
│   ├── security/              # Complete security implementation
│   │   ├── code_validator.py  # AST-based validation (3 levels)
│   │   ├── sandbox.py         # Resource-limited execution
│   │   └── __init__.py        # Security module exports
│   ├── models/                # Pydantic data models
│   ├── services/              # Business logic services  
│   ├── main_simple.py         # Production FastAPI server (DEPLOYED)
│   └── requirements.txt       # Complete dependencies
├── frontend/                   # React TypeScript application
│   ├── src/
│   │   ├── components/        # UI components + shadcn/ui
│   │   ├── services/          # API integration
│   │   └── types/             # TypeScript definitions
│   ├── package.json           # Frontend dependencies
│   └── vite.config.ts         # Build configuration
├── data_processing/           # Data analysis modules (COMPLETE)
│   ├── enhanced_data_cleaner.py    # Advanced preprocessing
│   ├── data_profiler.py            # Statistical analysis
│   ├── type_inference.py           # Column type detection
│   └── enhanced_preprocessor.py    # Data transformation
├── docker-compose.yml         # Multi-service orchestration
├── .gitignore                 # Comprehensive ignore rules
└── README.md                  # This documentation
```

## 🔧 Development

### Running the Backend Locally

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Start the development server
python main_simple.py
# Server will be available at http://localhost:8000
```

### Running Tests

```bash
# Security layer tests
python test_simple_security.py

# API endpoint tests  
python -m pytest backend/tests/ -v

# Frontend tests
cd frontend
npm test
```

### Development Workflow

1. **Backend Changes**: Edit files in `backend/`, server auto-reloads
2. **Security Testing**: Run security tests after any changes to validation logic
3. **API Testing**: Use http://localhost:8000/docs for interactive testing
4. **Frontend Integration**: Frontend connects to backend API automatically

### Code Quality Tools

```bash
# Python formatting
black backend/
isort backend/

# TypeScript checking  
cd frontend
npm run type-check
npm run lint
```

## 🔒 Security Implementation

Our security layer provides comprehensive protection for code execution:

### Code Validation (`code_validator.py`)
- **AST Analysis**: Static analysis of Python code structure
- **Security Levels**: STRICT (production), MODERATE (development), PERMISSIVE (testing)
- **Import Control**: Whitelist-based module restrictions
- **Function Blacklisting**: Prevents dangerous built-in functions
- **Risk Assessment**: Categorizes security threats

### Sandbox Environment (`sandbox.py`)
- **Resource Limits**: CPU time and memory restrictions
- **Execution Timeouts**: Prevents infinite loops
- **Platform Support**: Windows and Unix compatibility
- **Output Control**: Limits output size and format
- **Clean Namespace**: Isolated execution environment

### Validation Example
```python
from backend.security import CodeValidator, ValidationLevel

validator = CodeValidator(ValidationLevel.STRICT)
result = validator.validate("import pandas as pd\ndf = pd.read_csv('data.csv')")
# Returns: ValidationResult(is_valid=True, risks=[], violations=[])
```

## 📊 Usage Examples

### Basic Session and File Upload

```python
import requests

# Create a new analysis session
response = requests.post("http://localhost:8000/api/sessions")
session = response.json()
session_id = session["session_id"]

# Upload a data file
with open("sales_data.csv", "rb") as f:
    files = {"file": ("sales_data.csv", f, "text/csv")}
    upload_response = requests.post(
        f"http://localhost:8000/api/sessions/{session_id}/upload", 
        files=files
    )
print(f"Upload status: {upload_response.json()}")
```

### Secure Code Execution

```python
# Execute data analysis code securely
code_request = {
    "code": """
import pandas as pd
import matplotlib.pyplot as plt

# The uploaded file is automatically available as 'df'
print("Dataset shape:", df.shape)
print("Column info:")
print(df.info())

# Generate summary statistics
summary = df.describe()
print(summary)

# Create a simple visualization
plt.figure(figsize=(10, 6))
df.hist(bins=20)
plt.tight_layout()
plt.savefig('data_distribution.png')
plt.show()

# Return insights
correlation = df.corr()
print("Correlation matrix:")
print(correlation)
    """,
    "context_variables": {"max_rows": 1000}
}

response = requests.post(
    f"http://localhost:8000/api/sessions/{session_id}/execute", 
    json=code_request
)

result = response.json()
print("Execution output:", result["output"])
print("Generated files:", result.get("generated_files", []))
```

### Code Validation (Before Execution)

```python
# Validate code for security before execution
validation_request = {
    "code": "import os; os.system('rm -rf /')",  # Dangerous code
    "validation_level": "STRICT"
}

response = requests.post(
    f"http://localhost:8000/api/sessions/{session_id}/validate",
    json=validation_request
)

result = response.json()
if not result["is_valid"]:
    print("Code validation failed!")
    print("Security risks:", result["risks"])
    print("Violations:", result["violations"])
```

## 🔌 API Endpoints (Live at localhost:8000)

### Session Management
- `POST /api/sessions` - Create new analysis session
- `GET /api/sessions/{session_id}` - Get session information  
- `DELETE /api/sessions/{session_id}` - Delete session

### File Operations
- `POST /api/sessions/{session_id}/upload` - Upload data file
- `GET /api/sessions/{session_id}/files` - List uploaded files
- `GET /api/sessions/{session_id}/files/{filename}` - Download file

### Code Execution (Secure)
- `POST /api/sessions/{session_id}/execute` - Execute Python code securely
- `POST /api/sessions/{session_id}/validate` - Validate code without execution

### Analysis Operations  
- `POST /api/sessions/{session_id}/analyze` - Start comprehensive analysis
- `GET /api/sessions/{session_id}/results` - Get analysis results
- `GET /api/sessions/{session_id}/visualizations` - Get generated charts

### System Health
- `GET /health` - Application health check
- `GET /api/status` - Detailed system status

### Interactive Documentation
Visit http://localhost:8000/docs for live API testing interface.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add your feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review the troubleshooting guide in the wiki

---

**Built with ❤️ using LangGraph, FastAPI, and React**
