// AGI Astronaut Web Interface Script
// This interface can work in 2 modes:
// 1. Connected mode: Full functionality with local Ollama
// 2. Demo mode: Shows example responses without Ollama

const OLLAMA_URL = 'http://localhost:11434/api/generate';
const MODEL_NAME = 'space-agi-astronaut';

let isConnected = false;
let isDemoMode = false;

// Demo response templates
const demoResponses = {
    'navier-stokes': 'The Navier-Stokes equations describe fluid motion. For incompressible flow (∇·v = 0), the momentum equation is ∂v/∂t + (v·∇)v = -∇p/ρ + ν∇²v, where v is velocity, p is pressure, ρ is density, and ν is kinematic viscosity. These equations are fundamental to aerodynamics and spacecraft propulsion analysis.',
    'schwarzschild': 'The Schwarzschild radius for a 10 solar mass object is approximately 29.5 km. This is calculated using rs = 2GM/c² where G is the gravitational constant, M is the mass, and c is the speed of light. Any object compressed within this radius becomes a black hole.',
    'orbital': 'For orbital mechanics, we use Kepler\'s equations to calculate orbital parameters. The orbital period can be found using Kepler\'s third law: T² = (4π²/GM) * a³, where T is the period, G is gravitational constant, M is the central mass, and a is the semi-major axis.',
    'quantum': 'The Schrödinger equation (iℏ ∂ψ/∂t = Ĥψ) is fundamental to quantum mechanics. Its solutions give the wave function ψ, which contains all information about a quantum system. For the hydrogen atom, solutions yield energy levels En = -13.6 eV/n².',
    'relativity': 'Special relativity shows that E = mc², demonstrating the equivalence of mass and energy. For objects moving at relativistic speeds, we must account for time dilation (Δt = γΔt₀) and length contraction (L = L₀/γ), where γ = 1/√(1-v²/c²).',
};

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
            timeout: 5000,
        });

        if (response.ok) {
            isConnected = true;
            isDemoMode = false;
            statusElement.textContent = '✓ Connected to Ollama';
            statusElement.className = 'status connected';
        } else {
            throw new Error('Response not ok');
        }
    } catch (error) {
        isConnected = false;
        isDemoMode = true;
        statusElement.textContent = '⚠ Demo Mode (Local Ollama offline)';
        statusElement.className = 'status checking';
        console.log('Connection check failed, using demo mode:', error);
    }
}

async function sendMessage() {
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const message = userInput.value.trim();

    if (!message) return;

    // Disable input while processing
    userInput.disabled = true;
    sendButton.disabled = true;
    sendButton.textContent = 'Thinking...';

    // Add user message to chat
    addMessage('user', message);

    // Clear input
    userInput.value = '';

    try {
        let aiResponse;

        if (isConnected) {
            // Use real Ollama connection
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
            aiResponse = data.response || 'Sorry, I encountered an error processing your request.';
        } else {
            // Use demo mode with simulated responses
            aiResponse = generateDemoResponse(message);
        }

        // Add AI response to chat
        addMessage('system', aiResponse);

    } catch (error) {
        console.error('Error:', error);
        // Try demo mode if real connection fails
        const demoResponse = generateDemoResponse(message);
        addMessage('system', demoResponse);
    } finally {
        // Re-enable input
        userInput.disabled = false;
        sendButton.disabled = false;
        sendButton.textContent = 'Send';
        userInput.focus();
    }
}

function generateDemoResponse(userMessage) {
    const lowerMessage = userMessage.toLowerCase();

    // Check for keywords in demo responses
    for (const [keyword, response] of Object.entries(demoResponses)) {
        if (lowerMessage.includes(keyword)) {
            return response;
        }
    }

    // Generic fallback demo response
    if (lowerMessage.includes('orbit') || lowerMessage.includes('space')) {
        return 'I am an AGI Astronaut specialized in space exploration and advanced mathematics. I can help you with orbital mechanics, relativistic physics, quantum equations, and aerospace calculations. Currently running in demo mode. To use my full capabilities, please start Ollama with the space-agi-astronaut model.';
    }

    if (lowerMessage.includes('how') || lowerMessage.includes('what') || lowerMessage.includes('explain')) {
        return 'Great question! I\'m designed to provide detailed explanations of complex physical phenomena. For the most accurate responses, please connect to a local Ollama instance running the space-agi-astronaut model. In demo mode, I can show you example responses on topics like orbits, black holes, quantum mechanics, and relativity.';
    }

    if (lowerMessage.includes('solve') || lowerMessage.includes('calculate')) {
        return 'I can solve complex differential equations, orbital mechanics problems, and physics calculations. In demo mode, I\'m limited, but with a full connection to the AGI Astronaut model via Ollama, I can provide detailed step-by-step solutions to advanced problems in mathematics and physics.';
    }

    return 'Hello! I\'m your AGI Astronaut, ready to help with complex mathematical physics problems. Currently in demo mode - my responses are limited until you connect a local Ollama instance. To get started, run: ollama run space-agi-astronaut';
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
setInterval(checkConnection, 15000);