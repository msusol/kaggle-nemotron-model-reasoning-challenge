---
description: Guidelines for writing commit descriptions based on git diff and task context
globs: ["**/*"]
---

# Commit Description Guidelines

When writing commit messages, follow the [Conventional Commits](https://conventionalcommits.org/) specification. Analyze the git diff and task context to create meaningful descriptions that explain what changed and why.

## Structure

**Format:**
```
<type>[optional scope]: <description>

[optional body]

[optional footer]
```

- **type**: The type of change (feat, fix, docs, refactor, test, chore, etc.)
- **scope**: Optional context (e.g., component or file affected)
- **description**: Brief summary in imperative mood (50-72 characters recommended)
- **body**: Optional detailed explanation with bullet points for granular changes
- **footer**: Optional references to issues/PRs (e.g., "Closes #123")

**Examples:**
```
feat(queue-consumer): add OCI Queue consumer with mock implementation

- Create queue_consumer.py with polling logic
- Add mock QueueClient for local testing
- Implement message processing and error handling
```

```
fix(probe): correct liveness probe to prevent container restarts

- Change health check from import to simple Python execution
- Prevent accidental consumer startup during health checks
- Add readiness probe for proper startup verification
```

```
docs: document Argo Workflows integration

- Add integration section to README.md
- Include producer setup examples
- Document scaling and configuration options
```

## Analysis Process

### 1. Review Git Diff
- **New files**: Describe what functionality they add
- **Modified files**: Explain what changed and why
- **Deleted files**: Note what was removed and impact

### 2. Consider Task Context
- **Feature implementation**: Focus on new capabilities
- **Bug fixes**: Explain the problem and solution
- **Refactoring**: Describe improvements and standards applied
- **Documentation**: Highlight what guidance was added

### 3. Categorize Changes
- **feat**: New features or functionality
- **fix**: Bug fixes
- **docs**: Documentation changes
- **refactor**: Code restructuring without functional changes
- **test**: Test-related changes
- **chore**: Maintenance tasks (dependencies, config, etc.)

## Examples by File Type

### Docker/Kubernetes Files
- `Dockerfile`: "Add Python 3.12 base image with OCI SDK"
- `deployment.yaml`: "Configure Kubernetes deployment with health probes"
- `docker-compose.yml`: "Set up local development environment"

### Python Code Files
- `main.py`: "Implement core ingest logic with error handling"
- `queue_consumer.py`: "Add mock queue client for local testing"
- Requirements: "Update OCI SDK to latest version"

### Configuration Files
- `.clinerules/`: "Define Python coding standards and best practices"
- `README.md`: "Document deployment and integration procedures"

## Best Practices

- **Be specific**: Mention actual files and functions changed
- **Explain why**: Include context about the problem solved or feature added
- **Keep concise**: Aim for 50-72 characters in the subject line
- **Use imperative mood**: "Add feature" not "Added feature"
- **Reference issues**: Include issue/PR numbers when applicable

## Task-Based Examples

### For This Project (OCI Queue Consumer)
```
feat(docker): create OCI Queue consumer Docker image with mock client

- Build Python 3.12 container with OCI SDK
- Implement queue consumer with instance principals auth
- Add mock client for local development and testing
- Create main.py placeholder for ingest logic
```

```
feat(k8s): add Kubernetes deployment with proper namespacing

- Create argo-workflows-sharded namespace
- Configure deployment with health probes and resource limits
- Set up local registry for Rancher Desktop deployment
- Add environment variables for queue configuration
```

```
docs: document Argo Workflows integration and scaling

- Add integration flow and producer setup examples
- Document consumer configuration and environment variables
- Include scaling options with KEDA and horizontal pod autoscaling
- Provide troubleshooting guide for common issues
```

```
refactor: apply Python coding standards and type hints

- Add type hints to function parameters and return types
- Use single quotes for string literals consistently
- Implement proper import organization and grouping
- Add comprehensive docstrings following Google style
```

```
fix(probe): correct health probes to prevent container restarts

- Fix liveness probe to use simple health check
- Prevent accidental consumer startup during probe execution
- Add readiness probe for proper startup verification
- Test probe behavior with mock implementations
```
