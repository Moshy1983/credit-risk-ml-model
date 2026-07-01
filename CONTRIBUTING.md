# Contributing Guidelines

Thank you for your interest in improving this project! This is a portfolio showcase project, but contributions are welcome.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:
1. Check existing issues to avoid duplicates
2. Open a new issue with:
   - Clear description of the problem or suggestion
   - Steps to reproduce (for bugs)
   - Expected vs. actual behavior
   - Environment details (OS, Node version, Python version)

### Pull Requests

1. **Fork** the repository
2. **Create a branch** with a descriptive name:
   ```bash
   git checkout -b feature/add-roc-interaction
   git checkout -b fix/shap-waterfall-labels
   ```
3. **Make your changes** with clear, focused commits
4. **Test** your changes:
   ```bash
   # Python tests
   python model/train.py
   python model/evaluate.py
   
   # Frontend tests
   npm run lint
   npm run build
   ```
5. **Submit a PR** with:
   - Clear description of changes
   - Screenshots (for UI changes)
   - Link to related issue(s)

## Code Style

### Python
- Follow **PEP 8** guidelines
- Use **type hints** where possible
- Document functions with docstrings
- Keep functions focused and modular

### TypeScript / React
- Use **TypeScript** for all new components
- Follow **functional component** patterns with hooks
- Use **Tailwind CSS** utility classes for styling
- Maintain responsive design (mobile-first)

## Development Workflow

```bash
# 1. Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
npm install

# 2. Run Python pipeline
python model/generate_synthetic_data.py
python model/train.py
python model/evaluate.py
python model/shap_analysis.py

# 3. Run frontend dev server
npm run dev

# 4. Build for production
npm run build
```

## Areas for Contribution

- **Data**: Additional synthetic features, more realistic correlation matrices
- **Model**: Alternative architectures (LightGBM, CatBoost, neural nets)
- **Frontend**: Additional visualizations, dark mode, responsive improvements
- **Docs**: Translation, tutorials, video walkthroughs
- **Tests**: Unit tests for Python pipeline, component tests for React

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Reach out via GitHub issues or visit [mosef.dev](https://mosef.dev).
