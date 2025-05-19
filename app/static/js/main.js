async function sendMessage() {
  const input = document.getElementById('user-input');
  const message = input.value;
  if (!message) return;

  appendMessage("Bạn", message);

  const response = await fetch('/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ message: message })
  });

  const data = await response.json();
  appendMessage("Bot", data.response);

  input.value = '';
}

function appendMessage(sender, message) {
  const chatBox = document.getElementById('chat-box');
  const msg = document.createElement('div');
  msg.innerHTML = `<strong>${sender}:</strong> ${message}`;
  chatBox.appendChild(msg);
  chatBox.scrollTop = chatBox.scrollHeight;
}