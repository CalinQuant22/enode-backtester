Contributing Guide
=================

We welcome contributions to the Enode Backtester project! This guide will help you get started with contributing code, documentation, examples, and bug reports.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   .. code-block:: bash

       git clone https://github.com/your-username/enode-backtester.git
       cd enode-backtester

2. **Set Up Development Environment**

   .. code-block:: bash

       # Using uv (recommended)
       uv sync --dev
       
       # Or using pip
       python -m venv .venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate
       pip install -r requirements.txt
       pip install -e .

3. **Install Development Dependencies**

   .. code-block:: bash

       pip install pytest black flake8 mypy sphinx

4. **Verify Installation**

   .. code-block:: bash

       python -m pytest tests/
       python test_library.py

Types of Contributions
----------------------

Code Contributions
~~~~~~~~~~~~~~~~~~

**New Features**
  - Strategy templates and examples
  - Risk management rules
  - Performance metrics
  - Data handlers
  - Execution models

**Bug Fixes**
  - Fix existing issues
  - Improve error handling
  - Performance optimizations

**Improvements**
  - Code quality enhancements
  - Better error messages
  - Performance optimizations

Documentation
~~~~~~~~~~~~~

**User Documentation**
  - Tutorial improvements
  - Usage examples
  - Best practices guides
  - FAQ additions

**API Documentation**
  - Docstring improvements
  - Code examples
  - Parameter descriptions

**Examples**
  - Strategy implementations
  - Use case demonstrations
  - Integration examples

Testing
~~~~~~~

**Unit Tests**
  - Test new features
  - Improve test coverage
  - Edge case testing

**Integration Tests**
  - End-to-end workflows
  - Multi-component testing
  - Performance benchmarks

Development Guidelines
----------------------

Code Style
~~~~~~~~~~

We follow Python best practices and PEP 8:

.. code-block:: bash

    # Format code
    black backtester/ tests/
    
    # Check style
    flake8 backtester/ tests/
    
    # Type checking
    mypy backtester/

**Key Standards:**
  - Use type hints for all function parameters and return values
  - Write descriptive docstrings for all public methods
  - Keep functions focused and single-purpose
  - Use meaningful variable and function names
  - Follow existing code patterns and conventions

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

All contributions must include appropriate tests:

.. code-block:: python

    # Example test structure
    def test_new_feature():
        """Test description of what is being tested"""
        # Arrange
        setup_data = create_test_data()
        
        # Act
        result = new_feature(setup_data)
        
        # Assert
        assert result.expected_property == expected_value
        assert len(result.items) > 0

**Testing Guidelines:**
  - Write tests before implementing features (TDD encouraged)
  - Test both happy path and error conditions
  - Use descriptive test names
  - Include edge cases and boundary conditions
  - Maintain high test coverage (aim for >90%)

Documentation Standards
~~~~~~~~~~~~~~~~~~~~~~~

**Docstring Format:**

.. code-block:: python

    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
        """Calculate the Sharpe ratio for a series of returns.
        
        The Sharpe ratio measures risk-adjusted return by dividing excess return
        by the standard deviation of returns.
        
        Args:
            returns: List of periodic returns (e.g., daily, monthly)
            risk_free_rate: Risk-free rate for the same period (default: 0.0)
            
        Returns:
            Sharpe ratio as a float. Higher values indicate better risk-adjusted performance.
            
        Raises:
            ValueError: If returns list is empty or contains invalid values
            
        Example:
            >>> returns = [0.01, 0.02, -0.01, 0.03]
            >>> sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.001)
            >>> print(f"Sharpe ratio: {sharpe:.2f}")
        """

**Documentation Guidelines:**
  - Use clear, concise language
  - Include practical examples
  - Explain the "why" not just the "what"
  - Link to related concepts and functions
  - Keep examples up-to-date with API changes

Contribution Workflow
---------------------

1. **Create an Issue**
   
   Before starting work, create an issue to discuss:
   - Feature requirements and design
   - Bug reproduction steps
   - Implementation approach

2. **Create a Branch**

   .. code-block:: bash

       git checkout -b feature/your-feature-name
       # or
       git checkout -b fix/issue-description

3. **Implement Changes**
   
   - Write code following our style guidelines
   - Add comprehensive tests
   - Update documentation as needed
   - Ensure all tests pass

4. **Commit Changes**

   .. code-block:: bash

       git add .
       git commit -m "feat: add new risk management rule
       
       - Implement volatility-based position sizing
       - Add comprehensive tests
       - Update documentation with examples"

   **Commit Message Format:**
   - Use conventional commits format
   - Start with type: feat, fix, docs, test, refactor
   - Include clear description of changes
   - Reference issue numbers when applicable

5. **Run Quality Checks**

   .. code-block:: bash

       # Run tests
       python -m pytest tests/ -v
       
       # Check code style
       black --check backtester/ tests/
       flake8 backtester/ tests/
       
       # Type checking
       mypy backtester/
       
       # Test documentation build
       cd docs/
       make html

6. **Submit Pull Request**
   
   - Create a clear PR description
   - Link to related issues
   - Include testing instructions
   - Request appropriate reviewers

Pull Request Guidelines
-----------------------

PR Description Template
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Description
    Brief description of changes and motivation.
    
    ## Type of Change
    - [ ] Bug fix (non-breaking change that fixes an issue)
    - [ ] New feature (non-breaking change that adds functionality)
    - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
    - [ ] Documentation update
    
    ## Testing
    - [ ] Tests pass locally
    - [ ] New tests added for new functionality
    - [ ] Manual testing completed
    
    ## Checklist
    - [ ] Code follows project style guidelines
    - [ ] Self-review completed
    - [ ] Documentation updated
    - [ ] No breaking changes (or clearly documented)

Review Process
~~~~~~~~~~~~~~

1. **Automated Checks**: CI/CD pipeline runs tests and quality checks
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Reviewers test functionality manually if needed
4. **Approval**: At least one maintainer approval required
5. **Merge**: Squash and merge to main branch

Specific Contribution Areas
---------------------------

Strategy Examples
~~~~~~~~~~~~~~~~~

We especially welcome new strategy implementations:

.. code-block:: python

    class YourStrategy(BaseStrategy):
        """Brief description of strategy logic and use case"""
        
        def __init__(self, event_queue, data_handler, param1=default_value):
            super().__init__(event_queue, data_handler)
            self.param1 = param1
            # Initialize strategy state
        
        def on_stock_event(self, event: StockEvent) -> None:
            """Process market data and generate signals"""
            # Your strategy logic here
            pass

**Strategy Contribution Guidelines:**
  - Include clear documentation of strategy logic
  - Provide parameter explanations and sensible defaults
  - Add example usage and expected performance characteristics
  - Include appropriate risk management considerations

Risk Management Rules
~~~~~~~~~~~~~~~~~~~~~

Custom risk rules are valuable contributions:

.. code-block:: python

    class YourRiskRule(BaseRiskRule):
        """Description of what risk this rule manages"""
        
        def __init__(self, threshold_param=default_value):
            self.threshold_param = threshold_param
        
        def check(self, portfolio, signal_event, proposed_quantity, data_handler):
            """Validate order against risk criteria"""
            # Your risk logic here
            return RiskCheckResult(approved=True/False, reason="explanation")

Dashboard Enhancements
~~~~~~~~~~~~~~~~~~~~~~

Dashboard improvements are always welcome:

- New chart types and visualizations
- Additional performance metrics
- UI/UX improvements
- Mobile responsiveness enhancements
- Export functionality

Performance Metrics
~~~~~~~~~~~~~~~~~~~

New performance and risk metrics:

.. code-block:: python

    def calculate_your_metric(portfolio_data: Dict) -> float:
        """Calculate your custom performance metric
        
        Args:
            portfolio_data: Dictionary containing equity curve, trades, etc.
            
        Returns:
            Calculated metric value
        """
        # Your calculation logic
        return metric_value

Community Guidelines
--------------------

Code of Conduct
~~~~~~~~~~~~~~~

- Be respectful and inclusive
- Provide constructive feedback
- Help newcomers learn and contribute
- Focus on technical merit in discussions
- Maintain professional communication

Getting Help
~~~~~~~~~~~~

**For Questions:**
  - Create a GitHub issue with the "question" label
  - Join our community discussions
  - Check existing documentation and examples

**For Bugs:**
  - Search existing issues first
  - Provide minimal reproduction case
  - Include environment details
  - Add relevant error messages and logs

**For Feature Requests:**
  - Describe the use case and motivation
  - Suggest implementation approach if possible
  - Consider backward compatibility
  - Discuss with maintainers before large changes

Recognition
-----------

Contributors will be recognized in:

- Project README contributors section
- Release notes for significant contributions
- Documentation acknowledgments
- Community highlights

Thank you for contributing to the Enode Backtester project! Your contributions help make quantitative trading more accessible and powerful for everyone.