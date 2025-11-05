import React, { useState } from "react";

function App() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user", text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch("http://127.0.0.1:8000/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: input }),
      });

      const data = await response.json();
      const botMessage = { sender: "bot", text: data.answer || "No response" };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      setMessages((prev) => [...prev, { sender: "bot", text: "⚠️ Error connecting to backend" }]);
    }

    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.header}>🧠 Gemma Medical Chatbot</h2>
      <div style={styles.chatBox}>
        {messages.map((msg, i) => (
          <div
            key={i}
            style={{
              ...styles.message,
              alignSelf: msg.sender === "user" ? "flex-end" : "flex-start",
              backgroundColor: msg.sender === "user" ? "#d1f7c4" : "#f1f0f0",
            }}
          >
            <b>{msg.sender === "user" ? "You: " : "Gemma: "}</b>
            {msg.text}
          </div>
        ))}
        {loading && <div style={styles.loading}>Gemma is thinking...</div>}
      </div>

      <div style={styles.inputArea}>
        <input
          style={styles.input}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your medical question..."
        />
        <button style={styles.button} onClick={sendMessage}>
          Send
        </button>
      </div>
    </div>
  );
}

const styles = {
  container: { maxWidth: "600px", margin: "40px auto", fontFamily: "Arial, sans-serif" },
  header: { textAlign: "center" },
  chatBox: {
    border: "1px solid #ccc",
    borderRadius: "10px",
    padding: "10px",
    height: "400px",
    overflowY: "auto",
    display: "flex",
    flexDirection: "column",
    gap: "10px",
  },
  message: { padding: "8px 12px", borderRadius: "8px", maxWidth: "80%" },
  inputArea: { marginTop: "10px", display: "flex", gap: "10px" },
  input: { flex: 1, padding: "10px", borderRadius: "5px", border: "1px solid #aaa" },
  button: { padding: "10px 20px", borderRadius: "5px", backgroundColor: "#4caf50", color: "white", border: "none" },
  loading: { fontStyle: "italic", color: "#888" },
};

export default App;
