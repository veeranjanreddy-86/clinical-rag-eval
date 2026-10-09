# Security Policy

This is a portfolio/reference project. It ships only **synthetic** documents and must not be used
with real patient data without a proper security, privacy and compliance review.

- **Reporting:** please open a private security advisory on GitHub
  (Security tab -> "Report a vulnerability") rather than a public issue.
- **Secrets:** no credentials are stored in this repository. Provider keys are read from
  environment variables only (see `.env.example`); `.env` is gitignored.
- **PHI/PII:** the regex redaction in `guardrails.py` is a defense-in-depth safety net, not a
  certified de-identification method.
