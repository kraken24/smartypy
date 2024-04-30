# Build Your Own Python Assistant with Llama3 in Just 10 Minutes!

This repository provides necessary files and documentation to create your personal python coding assistant.

## Blog
A detailed blog with instructions can be found [here](./blog.md)

## Summary of Steps
1. Download Ollama
2. Download model to use (I am using llama3)
`ollama pull llama3`.
3. Create a system message for Llama3 to help with only python coding
    1. Create Modelfile
    ```bash
        # ./Modelfile
        FROM llama3
        PARAMETER temperature 0
        SYSTEM "You are Python coding assistant. Help me autocomplete my Python code."
    ```
    2. Create a model from modelfile with
    `ollama create my-python-assistant -f ./Modelfile`
4. Download Continue VS Code extension to work with LLMs [here](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
5. Connect VS Code with Ollama and the model you created.


## Reference
This blog was inspired from the LinkedIn post by Pau Bajo [here]( https://www.linkedin.com/posts/pau-labarta-bajo-4432074b_machinelearning-llmops-llms-activity-7190620863602282498-q1Lg?utm_source=share&utm_medium=member_desktop)
