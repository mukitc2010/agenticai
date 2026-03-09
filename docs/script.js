// AGI Astronaut Web Interface Script

const OLLAMA_URL = 'http://localhost:11434/api/generate';
const MODEL_NAME = 'space-agi-astronaut';

let isConnected = false;

// Check Ollama connection on load
window.addEventListener('load', () => {
    checkConnection();
    document.getElementById('user-input').focus();
});

// Check if Ollama is running
async function checkConnection() {
    const statusElement = document.getElementById('status');

    try {
        const response = await fetch(OLLAMA_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: MODEL_NAME,
                prompt: 'test',
                stream: false,
            }),
        });

        if (response.ok) {
            isConnected = true;
            statusElement.textContent = 'Connected to Ollama';
            statusElement.className = 'connected';
        } else {
            throw new Error('Response not ok');
        }
    } catch (error) {
        isConnected = false;
        statusElement.textContent = 'Disconnected - Start Ollama and load model';
        statusElement.className = 'disconnected';
        console.log('Connection check failed:', error);
    }
}

async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const message = userInput.value.trim();

    if (!message) return;

    if (!isConnected) {
        alert('Please ensure Ollama is running and the space-agi-astronaut model is loaded.\n\nRun: ollama run space-agi-astronaut');
        return;
    }

    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.textContent = 'Thinking...';

    // Add user message to chat
    addMessage('user', message);

    // Clear input
    userInput.value = '';

    try {
        // Send request to Ollama
        const response = await fetch(OLLAMA_URL, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model: MODEL_NAME,
                prompt: message,
                stream: false,
            }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        const aiResponse = data.response || 'Sorry, I encountered an error processing your request.';

        // Add AI response to chat
        addMessage('system', aiResponse);

    } catch (error) {
        console.error('Error:', error);
        addMessage('system', 'Error: Unable to connect to the AGI Astronaut model. Please ensure Ollama is running and the model is loaded.');
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        userInput.focus();
    }
}

function addMessage(type, content) {
    const chatMessages = document.getElementById('chat-messages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (type === 'user') {
        contentDiv.innerHTML = `<strong>You:</strong> ${content}`;
    } else {
        contentDiv.innerHTML = `<strong>AGI Astronaut:</strong> ${formatResponse(content)}`;
    }

    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);

    // Scroll to bottom
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function formatResponse(text) {
    // Basic formatting for math expressions and code
    return text
        .replace(/\n/g, '<br>')
        .replace(/\\\(/g, '$')
        .replace(/\\\)/g, '$')
        .replace(/\\\[/g, '$$')
        .replace(/\\\]/g, '$$');
}

// Allow sending message with Enter key
document.getElementById('user-input').addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
});

// Periodic connection check
setInterval(checkConnection, 10000);