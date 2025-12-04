nexusforgecompress-prototype/
├── LICENSE                 # MIT
├── README.md               # With disclaimer, roadmap, benchmarks
├── setup.py                # Updated
├── requirements.txt        # With versions, notes
├── CONTRIBUTING.md         # Guidelines
├── SECURITY.md             # Vuln reporting
├── CHANGELOG.md            # Semantic versions
├── nfc_prototype/          # Package
│   ├── __init__.py
│   ├── core.py             # Hardened with 64-bit, JSON, streaming; fixed zipnn usage
│   ├── utils.py            # Helpers
│   └── adapters/           # Placeholder
│       └── __init__.py
├── tests/                  # Expanded (fuzz-like, corruption)
│   ├── __init__.py
│   └── test_core.py
├── examples/               # Demo
│   └── example.py
├── bench/                  # Reproducible benchmarks
│   ├── download_model.sh   # Sample data
│   └── run_bench.py        # Fair comparisons
└── .gitignore              # Standard