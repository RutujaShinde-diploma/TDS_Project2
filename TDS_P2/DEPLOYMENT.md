# 🚀 Render Deployment Guide

## **Overview**
This guide explains how to deploy the Data Analyst Agent API to Render using Python runtime (no Docker).

## **Prerequisites**
- GitHub repository with the code
- Render account
- OpenAI API key

## **Step 1: Render Dashboard Setup**

### **Create New Web Service**
1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Select the repository and branch

### **Service Configuration**
- **Name**: `data-analyst-agent-api` (or your preferred name)
- **Environment**: `Python 3`
- **Region**: Choose closest to your users
- **Branch**: `main` (or your default branch)
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python render_startup.py`

## **Step 2: Environment Variables**

### **Required Variables**
Set these in Render dashboard → Environment:

| Variable | Value | Description |
|----------|-------|-------------|
| `OPENAI_API_KEY` | `sk-...` | Your OpenAI API key |
| `PORT` | `8000` | Port (Render sets this automatically) |

### **Optional Variables**
| Variable | Value | Description |
|----------|-------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `MAX_EXECUTION_TIME` | `300` | Max execution time in seconds |
| `MAX_FILE_SIZE` | `100000000` | Max file size in bytes |
| `SANDBOX_TIMEOUT` | `60` | Sandbox execution timeout |

## **Step 3: Deploy**

1. **Click "Create Web Service"**
2. **Wait for build to complete** (5-10 minutes)
3. **Check logs** for any errors
4. **Test the API** using the provided URL

## **Step 4: Testing**

### **Health Check**
```bash
curl https://your-service-name.onrender.com/health
```

### **Test File Upload**
```bash
curl -X POST https://your-service-name.onrender.com/api/ \
  -F "questions=@test_sample_sales/questions.txt" \
  -F "files=@test_sample_sales/sample-sales.csv"
```

## **Troubleshooting**

### **Common Issues**
1. **Build fails**: Check requirements.txt and Python version
2. **Import errors**: Ensure all packages are in requirements.txt
3. **Environment variables**: Verify all required vars are set
4. **File permissions**: Check workspace and logs directory creation

### **Logs**
- View logs in Render dashboard → Logs tab
- Check for Python errors and import issues
- Verify environment variable loading

## **Performance Notes**
- **Free tier**: 750 hours/month, sleeps after 15 minutes of inactivity
- **Paid tier**: Always-on, better performance
- **Cold starts**: First request may take 10-30 seconds

## **Security**
- API keys are encrypted in Render
- No Docker containers (simpler security model)
- File uploads are processed in memory
- Generated code runs in restricted Python environment

## **Support**
If you encounter issues:
1. Check Render logs
2. Verify environment variables
3. Test locally first
4. Check GitHub issues
