# TDS Project 2: Data Analyst Agent

## 🎯 Project Overview
A deployed data analyst agent that uses LLMs to source, prepare, analyze, and visualize any data. The system automatically processes user queries, generates analysis plans, and executes them to provide structured results within 5 minutes.

## ✨ Key Features
- **Intelligent Planning**: LLM-powered action planning based on user questions
- **Multi-Format Support**: Handles various data formats automatically
- **Dynamic Analysis**: Automatically detects task type and generates appropriate code
- **Progressive Output**: Builds results step-by-step through multiple actions
- **Flexible Output**: Supports both JSON object and array formats based on user requirements
- **Cache Management**: Efficient caching system for code generation and execution
- **5-Minute Response**: Guaranteed response time for all analysis tasks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Required packages (see requirements.txt)

### Installation
```bash
git clone https://github.com/RutujaShinde-diploma/TDS_Project2.git
cd TDS_Project2
pip install -r requirements.txt
```

### Running the Application
```bash
python main.py
```
The server will start on `http://localhost:8002`

## 🌐 Deployment
The application is deployed on **Render** and accessible via the public API endpoint.

## 📁 Supported File Types
The system can accept and process various file formats:
- **Text Files**: .txt, .md, .json
- **Data Files**: .csv, .xlsx, .xls, .tsv
- **Image Files**: .png, .jpg, .jpeg, .gif, .bmp
- **Document Files**: .pdf, .doc, .docx
- **Code Files**: .py, .js, .html, .css
- **Archive Files**: .zip, .tar, .gz

## 🏗️ Project Structure

### Main Application Files
```
TDS_P2/
├── main.py                 # Main application entry point and server setup
├── code_generator.py       # Core LLM integration and code execution engine
├── planner.py              # Intelligent action planning and task orchestration
├── orchestrator.py         # Task execution coordination and workflow management
├── models.py               # Data models, structures, and database schemas
├── config.py               # Configuration settings and environment variables
├── requirements.txt        # Python dependencies and package versions
└── .env.example           # Environment variables template (no secrets)
```

### Utility and Support Directories
```
TDS_P2/
├── utils/                  # Utility functions and helper modules
│   ├── __init__.py        # Package initialization
│   ├── simple_storage.py  # Basic storage and caching utilities
│   └── helpers.py         # Common helper functions
├── storage/                # Data storage and cache management
│   ├── __init__.py        # Storage package initialization
│   ├── cache.json         # Cache storage file
│   └── database.py        # Database connection and operations
└── tests/                  # Test cases and validation scripts
    ├── test2/             # Network analysis test cases
    │   ├── questions.txt  # Test questions
    │   ├── edges.csv      # Test data
    │   └── test_network_analysis.py
    └── test3/             # Weather data analysis test cases
        ├── questions.txt  # Test questions
        └── sample-weather.csv  # Test data
```

### Configuration and Documentation
```
TDS_P2/
├── .gitignore             # Git ignore patterns
├── README.md              # This documentation file
├── LICENSE                # MIT License file
└── .env.example          # Environment variables template
```

## 🔌 API Endpoints

### Main Analysis Endpoint
```bash
POST https://app.example.com/api/
```
- **Purpose**: Main data analysis endpoint
- **Parameters**: questions.txt (required) + optional data files
- **Response Time**: Within 5 minutes

### Cache Management Endpoints
```bash
POST https://app.example.com/api/cache/clear/code
```
- **Purpose**: Clear code generation cache

```bash
POST https://app.example.com/api/cache/clear/data
```
- **Purpose**: Clear data processing cache

## 📋 Implementation Guide

### For Developers Who Want to Implement

#### Step 1: Clone and Setup
```bash
git clone https://github.com/RutujaShinde-diploma/TDS_Project2.git
cd TDS_Project2
pip install -r requirements.txt
```

#### Step 2: Configure Environment
- Copy `.env.example` to `.env`
- Set up your LLM API keys in `.env`
- Configure timeout and other settings
- Set up your database/storage if needed

#### Step 3: Understand Core Components
- **main.py**: Entry point and server setup
- **code_generator.py**: LLM integration and code execution
- **planner.py**: Action planning and task orchestration
- **orchestrator.py**: Task execution coordination
- **models.py**: Data structures and models

#### Step 4: Customize for Your Use Case
- Modify the LLM prompts in `code_generator.py`
- Adjust the action types in `planner.py`
- Customize output formats in `orchestrator.py`

#### Step 5: Deploy
- Choose your hosting platform (Render, AWS, etc.)
- Set up environment variables
- Deploy using your preferred method

## 🔌 API Usage

### Request Format
```bash
curl "https://app.example.com/api/" \
  -F "questions.txt=@questions.txt" \
  -F "data.csv=@data.csv" \
  -F "image.png=@image.png"
```

### Required Parameters
- **questions.txt**: ALWAYS required - contains the analysis questions
- **Additional files**: Zero or more data files in various formats

### Response Format
- **Response Time**: Within 5 minutes
- **Format**: As specified in questions.txt (JSON object, array, or custom format)

## 🧠 How It Works

### 1. User Input
- Upload questions.txt with analysis requirements
- Optionally upload data files (CSV, images, etc.)

### 2. Intelligent Planning
- System analyzes questions and available data
- LLM generates optimal action plan
- Actions are sequenced for efficient execution

### 3. Progressive Execution
- Each action builds upon previous results
- Intermediate results stored progressively
- Final export generates structured output

### 4. Dynamic Output Format
- **JSON Object**: When specific keys are specified in questions
- **Array Format**: When generic numbered answers are requested
- **Fallback**: Generic answer1, answer2, etc. when no format specified

## 📝 Questions Format

### JSON Object Format
```txt
JSON object with keys:
- `metric_name`: data_type
- `another_metric`: data_type
```

### Array Format
```txt
1. First question about the data?
2. Second question about the data?
3. Third question about the data?
```

## ⚙️ Configuration

Key settings in `config.py`:
- **Timeout**: Action execution timeout (default: 300s)
- **Model**: LLM model configuration
- **Cache**: Cache management settings
- **Response Time**: Configured for 5-minute guarantee

## 🔍 Supported Analysis Types

- **Statistical Analysis**: Mean, median, correlation, etc.
- **Data Processing**: Filtering, grouping, aggregation
- **Custom Calculations**: User-defined metrics and formulas
- **Data Export**: Structured output in various formats
- **Data Visualization**: Charts, graphs, and visual representations

## 🚨 Performance Requirements

- **Response Time**: Maximum 5 minutes for any analysis task
- **Scalability**: Handles multiple concurrent requests
- **Reliability**: Robust error handling and fallback mechanisms
- **Efficiency**: Optimized code generation and execution

## 🔒 Security Best Practices

- **Never commit your .env file or any secrets to version control**
- **Use .env.example to share variable names (not values)**
- **Always use environment variables for API keys and passwords**
- **Implement proper authentication and authorization**
- **Validate and sanitize all user inputs**
- **Use HTTPS for all API communications**

## 🛠️ Troubleshooting

### Common Issues
1. **Timeout Errors**: Increase timeout in config.py
2. **Cache Issues**: Clear cache using API endpoints
3. **Import Errors**: Check generated code for missing imports

### Debug Mode
Enable detailed logging for troubleshooting:
```python
# In config.py
DEBUG = True
```

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

## 🙏 Acknowledgements
- **IIT Madras Online Degree** - Educational platform and guidance
- **Course content by S. Anand** - Expert instruction and curriculum
- **Discourse Community** - Community support and collaboration
