# Contributing to NexusForgeCompress (NFC)

Welcome! We are excited to have you contribute to the Neural File System.

## How to Contribute
1.  **Fork the repo** and create a feature branch.
2.  **Run the full test suite** before submitting any changes:
    ```bash
    # Run all versioned tests
    python tests/v010_test.py
    # ... up to
    python tests/v200_distributed_test.py
    ```
3.  **Benchmark new features** to ensure numerical efficiency:
    ```bash
    python bench/hard_proof_bench.py
    ```
4.  **Submit a Pull Request** with a clear description of your changes and why they are needed.

## Style & Standards
- Adhere to the existing code style (clean, modular, and well-documented).
- Ensure all new features are covered by tests.
- Update `CHANGELOG.md` and `README.md` if applicable.

## Community & Conduct
Be respectful and professional in all interactions. Report issues or request features via GitHub Issues.

Licensed under MIT.
