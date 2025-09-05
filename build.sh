#!/bin/bash

# Create .streamlit directory if it doesn't exist
mkdir -p .streamlit

# Create default config if it doesn't exist
if [ ! -f .streamlit/config.toml ]; then
    cat > .streamlit/config.toml << EOF
[server]
headless = true
enableCORS = false
enableXsrfProtection = true
port = \$PORT

[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F8F9FA"
textColor = "#262730"
font = "sans-serif"
EOF
fi

# Install Python dependencies
pip install -r requirements.txt
