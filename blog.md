# Build Your Own Python Assistant with Llama3 in Just 10 Minutes!

Hey there, fellow coder! Ever feel like you're playing a never-ending game of ping pong between your code editor and documentation while tackling those Python projects?

In this blog post, I'm here to walk you through the super fun process of creating your very own Python code assistant using Ollama. And the best part? We'll seamlessly integrate it into Visual Studio Code (VS Code) so you can breeze through your coding adventures like a pro!

Let's dive in!

#### Step 1: Download Ollama

Ollama is an open-source tool that allows you to create custom language models for various programming tasks and run them locally.
Here is the download [link](https://ollama.com/download).

#### Step 2: Download the Model

Choose the model you want to use for your Python code assistant. I am using llama3 (8B) parameter model. Download it using the command `ollama pull llama3`.

#### Step 3: Create a System Message for the LLM

To instruct Ollama to focus only on Python coding assistance, we need to create a system message. This message will guide Ollama to help with Python code autocompletion. Create a Modelfile with the following content:

```bash
# ./Modelfile
FROM llama3
PARAMETER temperature 0
SYSTEM "Hey llama3, I need your help with Python code completion. I want you to analyze my current code and suggest the most likely and accurate completions based on my query, context and best practices. If you need any additional information to complete the task, feel free to ask me."
```

Navigate to the Modelfile directory in your terminal and create a personalized Python assistant model using the command `ollama create my-python-assistant -f ./Modelfile`.

#### Step 4: Download the Continue VS Code Extension

To seamlessly integrate your python code assistant into VS Code, you'll need the Continue VS Code extension. Continue is an open-source extension for VSCode that helps you connect your VSCode editor with the LLMs you download with Ollama. Here is the download [link](https://marketplace.visualstudio.com/items?itemName=Continue.continue).

#### Step 5: Connect VS Code with Your Assistant

Launch VS Code and navigate to the settings. Search for "Continue" and configure the extension to connect to your Ollama model (my-python-assistant). Once configured, your Python code assistant will be ready to use within VS Code. Once it is installed, follow the readme to better organise the Continue extension with your VS Code interface.


#### Step 6: Let Your Code Assistant Work Its Magic!

Now that everything's all set up, it's time to let your Python code assistant shine and supercharge your coding experience in VS Code! Get ready for some serious coding efficiency as your assistant swoops in with intelligent suggestions and lightning-fast autocompletions. Let me walk you through two awesome methods for editing your code with this assistant: ![alt text](/images/image.png)

##### 1. Inline Editing with Cmd + I
By pressing Cmd + I, you can instruct the LLM on what changes to make. Here is one simple example: ![Inline Editing Example](/images/image-1.png)
Your assistant will suggest before and after code options, giving you the power to choose the best fit for your project. ![Inline Editing Example Results](/images/image-2.png)

##### 2. Continue Interface Editing with Cmd + L
By pressing Cmd + L, the Continue Extension opens up a handy side window for interactive LLM-assisted code editing. Similar to inline editing, you can provide specific instructions, and your assistant will take care of the rest. ![Continue Interface Editing](/images/image-3.png)
The code output in the interface window even lets you input follow-up instructions, keeping your coding flow smooth and seamless. ![Continue Interface Editing Results](/images/image-4.png)

**Feel free to try out a different model or optimize the system prompt to deliver customized results for your use-case or a different programming language.**

### Reference:

This blog was inspired from the LinkedIn post by Pau Bajo [here]( https://www.linkedin.com/posts/pau-labarta-bajo-4432074b_machinelearning-llmops-llms-activity-7190620863602282498-q1Lg?utm_source=share&utm_medium=member_desktop)