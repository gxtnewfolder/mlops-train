# MLOps Training Project

A comprehensive MLOps training project demonstrating end-to-end machine learning workflows using Azure Machine Learning (Azure ML) CLI v2, Azure DevOps pipelines, and modern MLOps practices.

## 🎯 Project Overview

This project showcases a complete MLOps pipeline for training and deploying machine learning models using Azure Machine Learning. It includes:

- **Data Science Workflows**: Data preprocessing, model training, and evaluation
- **MLOps Infrastructure**: Azure ML workspace setup, compute resources, and model deployment
- **CI/CD Pipelines**: Azure DevOps pipelines for automated training and deployment
- **Monitoring & Observability**: MLflow tracking and Azure Monitor integration
- **Best Practices**: Code quality tools, testing, and security considerations

## 🏗️ Architecture

The project is organized into several key components:

```
mlops-train/
├── data-science/          # Data science workflows and experiments
│   ├── src/              # Source code for data processing and training
│   ├── environment/      # Conda environment configurations
│   └── experiment/       # MLflow experiments and tracking
├── mlops/                # MLOps infrastructure and pipelines
│   ├── azureml/         # Azure ML workspace configurations
│   └── devops-pipelines/ # Azure DevOps pipeline definitions
├── infra/               # Infrastructure as Code (IaC) templates
├── aml-cli-v2/          # Azure ML CLI v2 scripts and configurations
├── data/                # Data storage and datasets
└── config-infra-prod.yml # Production environment configuration
```

## 🚀 Getting Started

### Prerequisites

- Azure subscription with appropriate permissions
- Azure CLI installed and configured
- Azure DevOps organization and project
- Python 3.8 or higher
- Conda or Miniconda

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd mlops-train
   ```

2. **Set up the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate azureml-cli-v2
   ```

3. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Azure CLI and Azure ML CLI v2**
   ```bash
   az login
   az extension add -n ml
   ```

### Environment Setup

1. **Create Azure Resources**
   ```bash
   # Create resource group
   az deployment group create --template-file infra/create-resource-group.yml
   
   # Create Azure ML workspace
   az deployment group create --template-file infra/create-workspace.yml
   
   # Create model registry
   az deployment group create --template-file infra/create-registry.yml
   ```

2. **Connect to Azure ML workspace**
   ```bash
   az ml workspace connect --workspace-name <workspace-name> --resource-group <resource-group>
   ```

## 📊 Data Science Workflow

### Data Preprocessing

The project includes a data preprocessing pipeline that:

- Reads raw data from various sources
- Performs data validation and quality checks
- Splits data into train/validation/test sets
- Logs data quality metrics using MLflow
- Supports monitoring integration with Azure Data Explorer

```bash
python data-science/src/prep.py \
    --raw_data <path-to-raw-data> \
    --train_data <output-train-path> \
    --val_data <output-val-path> \
    --test_data <output-test-path> \
    --enable_monitoring true
```

### Model Training

The training pipeline features:

- **Random Forest Regression** for cost prediction
- **Hyperparameter tuning** with configurable parameters
- **MLflow integration** for experiment tracking
- **Azure Monitor** for metrics collection
- **Model performance visualization**

```bash
python aml-cli-v2/train.py \
    --train_data <train-data-path> \
    --model_output <model-output-path> \
    --regressor__n_estimators 500 \
    --regressor__max_depth 10
```

### Model Evaluation

The evaluation process includes:

- R² score, MSE, RMSE, and MAE metrics
- Performance visualization (scatter plots)
- Model artifact logging
- Azure Monitor metric export

## 🔄 MLOps Pipelines

### Azure DevOps Pipelines

The project includes several Azure DevOps pipelines:

- **Infrastructure Pipeline**: Sets up Azure ML workspace and resources
- **Training Pipeline**: Automated model training and evaluation
- **Deployment Pipeline**: Model deployment to online/batch endpoints
- **Unit Testing Pipeline**: Code quality and testing automation

### Pipeline Features

- **Automated triggers** on code changes
- **Environment-specific configurations** (dev, staging, prod)
- **Security scanning** and compliance checks
- **Artifact management** and versioning
- **Rollback capabilities**

## 🛠️ Development

### Code Quality

The project enforces code quality standards:

- **Black**: Code formatting
- **Flake8**: Linting
- **isort**: Import sorting
- **pre-commit**: Git hooks for quality checks

### Testing

```bash
# Run unit tests
python -m pytest aml-cli-v2/tests/

# Run linting
flake8 .
black --check .
isort --check-only .
```

### Local Development

1. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

2. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make changes and commit**
   ```bash
   git add .
   git commit -m "Add your feature description"
   ```

## 🚀 Deployment

### Model Deployment

The project supports both online and batch deployments:

1. **Register the model**
   ```bash
   az ml model create --name <model-name> --path <model-path>
   ```

2. **Create deployment endpoint**
   ```bash
   az ml online-endpoint create --name <endpoint-name>
   ```

3. **Deploy the model**
   ```bash
   az ml online-deployment create --name <deployment-name> --endpoint <endpoint-name> --model <model-name>
   ```

### Environment Configuration

The project uses environment-specific configurations:

- **Development**: Local development and testing
- **Staging**: Pre-production validation
- **Production**: Live deployment with monitoring

## 📈 Monitoring and Observability

### MLflow Tracking

- Experiment tracking and comparison
- Model versioning and lineage
- Performance metrics logging
- Artifact management

### Azure Monitor Integration

- Custom metrics collection
- Application performance monitoring
- Alert configuration
- Dashboard creation

## 🔒 Security

### Security Features

- **Azure Key Vault** integration for secrets management
- **Managed Identity** for secure authentication
- **Network security** with private endpoints
- **Data encryption** at rest and in transit
- **Access control** with Azure RBAC

### Compliance

- **SOC 2** compliance considerations
- **GDPR** data protection measures
- **Audit logging** and monitoring
- **Security scanning** in CI/CD pipelines

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Code of Conduct

This project adheres to our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it before contributing.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Documentation

- [Azure Machine Learning Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)
- [Azure ML CLI v2 Reference](https://docs.microsoft.com/en-us/azure/machine-learning/reference-azure-machine-learning-cli)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)

### Issues and Questions

- Create an issue for bugs or feature requests
- Check existing issues for solutions
- Review the documentation for common questions

## 🗺️ Roadmap

- [ ] Multi-model deployment support
- [ ] Advanced monitoring dashboards
- [ ] Automated model retraining
- [ ] A/B testing framework
- [ ] Model explainability integration
- [ ] Cost optimization features

---

**Note**: This project is for educational and training purposes. For production use, ensure proper security configurations and compliance with your organization's policies.