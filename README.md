# Build Your Own Python Assistant with Llama3 in Just 10 Minutes!

This repository provides necessary files and documentation to create your personal python coding assistant in VS Code.

Alternatively, you can also create a ChatGPT like chatbot using fully local and secure setup. Read the guide below 🔽

## Blog

A detailed blog with explanations can be found [here](./blog.md) or [medium]()

## Summary of Steps: Python Code Assistant

1. Download & Install Ollama [here](https://ollama.com/download)
2. Download model to use (I am using llama3)
`ollama pull llama3`
3. Create a system message for Llama3 to help with only python coding
    1. Create Modelfile
    ```bash
        # ./Modelfile
        FROM llama3
        PARAMETER temperature 0
        SYSTEM "Hey llama3, I need your help with Python code completion. I want you to analyze my current code and suggest the most likely and accurate completions based on my query, context and best practices. If you need any additional information to complete the task, feel free to ask me."
    ```
    2. Create a model from modelfile with
    `ollama create my-python-assistant -f ./Modelfile`
4. Download Continue VS Code extension to work with LLMs [here](https://marketplace.visualstudio.com/items?itemName=Continue.continue)
5. Connect VS Code with Ollama and the model you created.


## Web-based Chatbot using local LLM (In 5 Steps)
To create a web-based chatbot
1. Download & Install Ollama [here](https://ollama.com/download)
    1. Open `http://localhost:11434` in web browser and you should see *Ollama is running* message
2. Download model to use (I am using llama3)
`ollama pull llama3`
3. Download & Install Docker [here](https://docs.docker.com/engine/install/)
4. In the terminal (zsh / terminator / powershell), run the following command:
    ```bash
    docker run -d \
    -p 3000:8080 \
    --add-host=host.docker.internal:host-gateway \
    -v open-webui:/app/backend/data \
    --name open-webui \
    --restart always \
    ghcr.io/open-webui/open-webui:main
    ```
5. Once this command is executed, you can access the Open Web UI by navigating to `http://localhost:3000` in your web browser.
6. Login Screen
Open Web UI will ask to create an account for the first login. As this app is running locally via Docker and storage is also local, the login credentials are stored persistently and can be retrieved automatically when the container is restarted. This also means that the login credentials are secured locally.
![alt text](/images/open_webui_login.png)
7. Model Selection & Chat
To start chatting with this chatbot, you need to select a model that you would like to use. You can download various models using `ollama pull <model-name>`. Below is the ChatGPT like UI for the chatbot. 
    1. Select use-case specific model for generatoin
    1. Upload files
    2. Read aloud
    3. Speech-to-text
![alt text](/images/open_webui_chat.png)

## Example
In the [examples](examples/) folder, there are some examples of llama3 capabilities and how it can massively improve your **efficiency** while coding. Since the temperature value in Modelfile is set to 0, the results should be reproducible.
- [x] Integrated code generation
- [x] Doc-Strings and Type Hinting
- [x] Writing unit tests

## Next Steps
- [ ] Optimize system prompt to include specific coding style, doc-string format
- [ ] Test the coding agent for other languages (JavaScript)


## Reference Links
This blog was inspired from the LinkedIn post by Pau Bajo [here]( https://www.linkedin.com/posts/pau-labarta-bajo-4432074b_machinelearning-llmops-llms-activity-7190620863602282498-q1Lg?utm_source=share&utm_medium=member_desktop)
