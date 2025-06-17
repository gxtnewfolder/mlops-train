# Installation and Configuration Guide

This guide provides detailed step-by-step instructions for setting up the MLOps Training Project on your local machine and in the Azure cloud environment.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Local Development Setup](#local-development-setup)
- [Azure Environment Setup](#azure-environment-setup)
- [Azure DevOps Configuration](#azure-devops-configuration)
- [Workspace Configuration](#workspace-configuration)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **Memory**: Minimum 8GB RAM (16GB recommended)
- **Storage**: At least 10GB free disk space
- **Network**: Stable internet connection for Azure services

### Required Accounts and Subscriptions

1. **Azure Subscription**
   - Active Azure subscription with billing enabled
   - Owner or Contributor permissions
   - [Create Azure Account](https://azure.microsoft.com/en-us/free/)

2. **Azure DevOps Organization**
   - Azure DevOps organization
   - Project creation permissions
   - [Create Azure DevOps Organization](https://dev.azure.com/)

3. **GitHub Account** (Optional)
   - For source code management
   - [Create GitHub Account](https://github.com/)

## 💻 Local Development Setup

### Step 1: Install Required Software

#### 1.1 Install Python and Conda

**Windows:**
```bash
# Download and install Miniconda
# Visit: https://docs.conda.io/en/latest/miniconda.html
# Download the Windows 64-bit installer

# Verify installation
conda --version
python --version
```

**macOS:**
```bash
# Using Homebrew
brew install miniconda

# Or download from official site
# Visit: https://docs.conda.io/en/latest/miniconda.html

# Verify installation
conda --version
python --version
```

**Ubuntu/Debian:**
```bash
# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Verify installation
conda --version
python --version
```

#### 1.2 Install Azure CLI

**Windows:**
```bash
# Download and install from Microsoft
# Visit: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli-windows

# Or using winget
winget install Microsoft.AzureCLI
```

**macOS:**
```bash
# Using Homebrew
brew install azure-cli

# Or using curl
curl -sL https://aka.ms/InstallAzureCLIDeb | bash
```

**Ubuntu/Debian:**
```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Verify installation
az --version
```

#### 1.3 Install Git

**Windows:**
```bash
# Download from https://git-scm.com/download/win
# Or using winget
winget install Git.Git
```

**macOS:**
```bash
# Using Homebrew
brew install git

# Or download from https://git-scm.com/download/mac
```

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install git
```

### Step 2: Clone and Setup Project

#### 2.1 Clone Repository
```bash
# Clone the repository
git clone <your-repository-url>
cd mlops-train

# Verify the structure
ls -la
```

#### 2.2 Create Conda Environment
```bash
# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate azureml-cli-v2

# Verify activation
conda info --envs
```

#### 2.3 Install Development Dependencies
```bash
# Install Python development dependencies
pip install -r requirements.txt

# Verify installations
python -c "import mlflow; print('MLflow version:', mlflow.__version__)"
python -c "import azureml; print('Azure ML version:', azureml.__version__)"
```

#### 2.4 Install Azure ML CLI Extension
```bash
# Install Azure ML CLI v2 extension
az extension add -n ml

# Verify installation
az ml --help
```

### Step 3: Configure Pre-commit Hooks
```bash
# Install pre-commit hooks
pre-commit install

# Verify installation
pre-commit --version
```

## ☁️ Azure Environment Setup

### Step 1: Azure Authentication

#### 1.1 Login to Azure
```bash
# Login to Azure
az login

# Verify login
az account show
```

#### 1.2 Set Default Subscription
```bash
# List available subscriptions
az account list --output table

# Set default subscription
az account set --subscription "<subscription-id>"

# Verify default subscription
az account show
```

### Step 2: Create Resource Group

#### 2.1 Create Resource Group
```bash
# Create resource group
az group create \
    --name "mlops-training-rg" \
    --location "southeastasia" \
    --tags "Environment=Training" "Project=MLOps"

# Verify resource group creation
az group show --name "mlops-training-rg"
```

#### 2.2 Update Configuration
Edit `config-infra-prod.yml`:
```yaml
# Update these values
resource_group: mlops-training-rg
location: southeastasia
environment: training
```

### Step 3: Deploy Azure ML Workspace

#### 3.1 Deploy Workspace Infrastructure
```bash
# Deploy Azure ML workspace
az deployment group create \
    --resource-group "mlops-training-rg" \
    --template-file infra/create-workspace.yml \
    --parameters workspaceName="mlops-training-ws" \
    --parameters location="southeastasia"

# Verify workspace creation
az ml workspace show --name "mlops-training-ws" --resource-group "mlops-training-rg"
```

#### 3.2 Deploy Model Registry
```bash
# Deploy model registry
az deployment group create \
    --resource-group "mlops-training-rg" \
    --template-file infra/create-registry.yml \
    --parameters registryName="mlops-training-registry"

# Verify registry creation
az ml registry show --name "mlops-training-registry" --resource-group "mlops-training-rg"
```

### Step 4: Configure Compute Resources

#### 4.1 Create Compute Cluster
```bash
# Create compute cluster for training
az ml compute create \
    --name "training-cluster" \
    --type "amlcompute" \
    --min-instances 0 \
    --max-instances 2 \
    --vm-size "STANDARD_DS11_v2" \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"
```

#### 4.2 Create Compute Instance (Optional)
```bash
# Create compute instance for development
az ml compute create \
    --name "dev-instance" \
    --type "computeinstance" \
    --vm-size "STANDARD_DS3_v2" \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"
```

### Step 5: Connect to Workspace
```bash
# Connect to Azure ML workspace
az ml workspace connect \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"

# Verify connection
az ml workspace show
```

## 🔄 Azure DevOps Configuration

### Step 1: Create Azure DevOps Project

#### 1.1 Create Project
1. Go to [Azure DevOps](https://dev.azure.com/)
2. Click "New Project"
3. Fill in project details:
   - **Project name**: `mlops-training`
   - **Description**: MLOps Training Project
   - **Visibility**: Private
   - **Version control**: Git
   - **Work item process**: Agile

#### 1.2 Configure Repository
```bash
# Add Azure DevOps as remote
git remote add azure https://dev.azure.com/<organization>/mlops-training/_git/mlops-training

# Push code to Azure DevOps
git push -u azure main
```

### Step 2: Create Service Connections

#### 2.1 Azure Resource Manager Service Connection
1. Go to Project Settings → Service Connections
2. Click "New Service Connection"
3. Select "Azure Resource Manager"
4. Configure:
   - **Scope level**: Subscription
   - **Subscription**: Select your subscription
   - **Resource Group**: `mlops-training-rg`
   - **Service connection name**: `AZURE-ARM-AML`

#### 2.2 Azure ML Workspace Service Connection
1. Create another service connection
2. Select "Azure Resource Manager"
3. Configure:
   - **Scope level**: Subscription
   - **Subscription**: Select your subscription
   - **Resource Group**: `mlops-training-rg`
   - **Service connection name**: `AZURE-ARM-AML-WS`

### Step 3: Configure Pipeline Variables

#### 3.1 Set Pipeline Variables
1. Go to Pipelines → Library
2. Create variable group: `mlops-training-variables`
3. Add variables:
   ```
   workspaceName: mlops-training-ws
   resourceGroup: mlops-training-rg
   location: southeastasia
   environment: training
   ```

#### 3.2 Configure Pipeline Permissions
1. Go to Project Settings → Pipelines → Settings
2. Enable "Allow public projects"
3. Configure build permissions for service connections

## ⚙️ Workspace Configuration

### Step 1: Configure MLflow Tracking

#### 1.1 Set MLflow Tracking URI
```bash
# Set MLflow tracking URI to Azure ML workspace
export MLFLOW_TRACKING_URI=$(az ml workspace show --name "mlops-training-ws" --resource-group "mlops-training-rg" --query "mlflow_tracking_uri" -o tsv)

# Verify MLflow configuration
mlflow ui --port 5000
```

#### 1.2 Configure MLflow Authentication
```bash
# Login to Azure ML workspace
az ml workspace connect --workspace-name "mlops-training-ws" --resource-group "mlops-training-rg"

# Verify MLflow can connect
python -c "import mlflow; mlflow.set_tracking_uri('azureml://'); print('MLflow connected successfully')"
```

### Step 2: Configure Data Storage

#### 2.1 Create Data Store
```bash
# Create blob storage data store
az ml datastore create-blob \
    --name "training-data" \
    --account-name "<storage-account-name>" \
    --container-name "data" \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"
```

#### 2.2 Register Datasets
```bash
# Register training dataset
az ml data create \
    --name "training-data" \
    --path "azureml://datastores/training-data/paths/data/" \
    --type "uri_folder" \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"
```

### Step 3: Configure Environment

#### 3.1 Create Conda Environment
```bash
# Create conda environment file
cat > environment.yml << EOF
name: mlops-training
channels:
  - conda-forge
  - defaults
dependencies:
  - python==3.8
  - pip
  - pip:
    - azureml-core
    - azureml-dataset-runtime[fuse]
    - mlflow
    - scikit-learn==0.24.2
    - pandas
    - numpy
    - matplotlib
EOF

# Register environment
az ml environment create \
    --name "mlops-training-env" \
    --conda-file environment.yml \
    --workspace-name "mlops-training-ws" \
    --resource-group "mlops-training-rg"
```

## 🧪 Testing Your Setup

### Step 1: Run Local Tests
```bash
# Run unit tests
python -m pytest aml-cli-v2/tests/ -v

# Run linting
flake8 .
black --check .
isort --check-only .

# Run pre-commit hooks
pre-commit run --all-files
```

### Step 2: Test Azure ML Connection
```bash
# Test workspace connection
python -c "
from azureml.core import Workspace
ws = Workspace.from_config()
print(f'Connected to workspace: {ws.name}')
print(f'Resource group: {ws.resource_group}')
print(f'Location: {ws.location}')
"
```

### Step 3: Test MLflow Integration
```bash
# Test MLflow tracking
python -c "
import mlflow
mlflow.set_tracking_uri('azureml://')
with mlflow.start_run():
    mlflow.log_param('test_param', 'test_value')
    mlflow.log_metric('test_metric', 0.95)
print('MLflow tracking test successful')
"
```

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Azure CLI Authentication Problems
```bash
# Clear cached credentials
az account clear

# Re-authenticate
az login

# Check subscription
az account show
```

#### Issue 2: Conda Environment Creation Fails
```bash
# Update conda
conda update conda

# Clean conda cache
conda clean --all

# Try creating environment again
conda env create -f environment.yml --force
```

#### Issue 3: Azure ML CLI Extension Issues
```bash
# Remove and reinstall extension
az extension remove -n ml
az extension add -n ml

# Update all extensions
az extension update -n ml
```

#### Issue 4: MLflow Connection Issues
```bash
# Check MLflow tracking URI
echo $MLFLOW_TRACKING_URI

# Set tracking URI manually
export MLFLOW_TRACKING_URI="azureml://"

# Test connection
python -c "import mlflow; mlflow.set_tracking_uri('azureml://'); print('Connected')"
```

#### Issue 5: Permission Issues
```bash
# Check Azure role assignments
az role assignment list --assignee $(az account show --query user.name -o tsv)

# Add Contributor role if needed
az role assignment create \
    --assignee $(az account show --query user.name -o tsv) \
    --role "Contributor" \
    --resource-group "mlops-training-rg"
```

### Getting Help

1. **Check Logs**: Review Azure ML workspace logs in the Azure portal
2. **Azure Documentation**: [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
3. **Community Support**: [Azure ML Community](https://techcommunity.microsoft.com/t5/azure-machine-learning/bd-p/AzureMachineLearning)
4. **GitHub Issues**: Create an issue in the project repository

## ✅ Verification Checklist

- [ ] Python 3.8+ installed and working
- [ ] Conda environment created and activated
- [ ] Azure CLI installed and authenticated
- [ ] Azure ML CLI extension installed
- [ ] Git installed and configured
- [ ] Pre-commit hooks installed
- [ ] Azure subscription configured
- [ ] Resource group created
- [ ] Azure ML workspace deployed
- [ ] Model registry created
- [ ] Compute resources configured
- [ ] Azure DevOps project created
- [ ] Service connections configured
- [ ] MLflow tracking configured
- [ ] Data storage configured
- [ ] Environment registered
- [ ] All tests passing
- [ ] Local development working

## 🚀 Next Steps

After completing the installation and configuration:

1. **Run Your First Pipeline**: Execute the training pipeline
2. **Deploy a Model**: Deploy your first model to an endpoint
3. **Set Up Monitoring**: Configure Azure Monitor and Application Insights
4. **Customize Workflows**: Adapt the pipelines for your specific use case
5. **Team Collaboration**: Set up team development workflows

---

**Note**: This guide assumes you have administrative access to your Azure subscription. If you're working in a corporate environment, you may need to coordinate with your IT department for certain permissions and configurations. 