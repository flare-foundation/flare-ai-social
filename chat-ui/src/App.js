import React from 'react';
import ChatInterface from './components/ChatInterface';
import './App.css';
import { Shield, Cpu, Code } from 'lucide-react';

function App() {
  return (
      <div className="app-container">
        <div className="sidebar">
          <div className="logo-container">
            <div className="logo">PH</div>
            <h2>Pugo Hilion</h2>
          </div>

          <div className="features-container">
            <h3>Key Features</h3>

            <div className="feature">
              <div className="feature-icon">
                <Shield size={20} />
              </div>
              <div className="feature-text">
                <h4>Secure AI Execution</h4>
                <p>Trusted Execution Environment (TEE) with remote attestation support</p>
              </div>
            </div>

            <div className="feature">
              <div className="feature-icon">
                <Code size={20} />
              </div>
              <div className="feature-text">
                <h4>Built-in Chat UI</h4>
                <p>Interact with your AI via a TEE-served interface</p>
              </div>
            </div>

            <div className="feature">
              <div className="feature-icon">
                <Cpu size={20} />
              </div>
              <div className="feature-text">
                <h4>Gemini Fine-Tuning</h4>
                <p>Fine-tune foundational models with custom datasets</p>
              </div>
            </div>
          </div>

          <div className="sidebar-footer">
            <p>© 2025 Pugo Hilion</p>
          </div>
        </div>

        <div className="main-content">
          <ChatInterface />
        </div>
      </div>
  );
}

export default App;