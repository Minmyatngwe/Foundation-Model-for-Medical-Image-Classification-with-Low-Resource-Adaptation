{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Chained Python Scripts",
      "type": "shell",
      "command": "python3 pipeline/adapt_with_lora.py && python3 pipeline/predict_and_explain.py",
      "group": {
        "kind": "build",
        "isDefault": true
      },
      "presentation": {
        "reveal": "always",
        "panel": "new"
      },
      "problemMatcher": []
    }
  ]
}