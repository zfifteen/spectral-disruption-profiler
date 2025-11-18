# Architecture

## Overview

The Spectral Disruption Profiler is a cloud-based SaaS platform built on a microservices architecture to ensure scalability, modularity, and high availability.

## Components

### Frontend
- **Framework**: React.js with TypeScript
- **Features**: User dashboard, sequence upload, visualization of results, API key management
- **Deployment**: Hosted on Vercel or AWS Amplify

### Backend
- **API Gateway**: AWS API Gateway or similar for RESTful endpoints
- **Microservices**:
  - **Sequence Processing Service**: Handles encoding and FFT computations
  - **Scoring Service**: Applies Z-invariant metrics and bootstrap CI
  - **Visualization Service**: Generates plots and reports
- **Framework**: FastAPI (Python) for performance and integration with scientific libraries
- **Database**: PostgreSQL for user data, MongoDB for sequence results
- **Caching**: Redis for session and result caching

### Core Algorithms
- **Encoding**: Complex waveform mapping (A=1+0i, etc.)
- **Phase Weighting**: θ′(n,k) with golden ratio φ
- **FFT**: SciPy-based fast Fourier transform
- **Scoring**: Composite disruption score with entropy, Δf₁, sidelobes

### Cloud Infrastructure
- **Provider**: AWS
- **Compute**: ECS/Fargate for containers, Lambda for serverless functions
- **Storage**: S3 for uploads, EFS for persistent data
- **Security**: VPC, IAM, encryption at rest/transit, HIPAA compliance

## Data Flow

1. User uploads sequences via frontend
2. API Gateway routes to Sequence Processing Service
3. Processed data sent to Scoring Service for metrics
4. Results stored and visualized
5. User downloads reports or accesses via API

## Scalability

- Horizontal scaling via Kubernetes
- Auto-scaling based on load
- Batch processing for large datasets

## Monitoring

- CloudWatch for logs and metrics
- Sentry for error tracking
- Performance monitoring for FFT computations