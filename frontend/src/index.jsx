import React from 'react';
import ReactDOM from 'react-dom/client';
import ChatWidget from './components/ChatWidget';
import './index.css';

window.IlariaChatbot = {
  init: (config = {}) => {
    const container = document.createElement('div');
    container.id = 'ilaria-chatbot-root';
    container.style.cssText = 'position: fixed; z-index: 9999;';
    document.body.appendChild(container);

    const root = ReactDOM.createRoot(container);
    root.render(
      <ChatWidget
        apiUrl={config.apiUrl || import.meta.env.VITE_API_URL || 'http://localhost:3001/api'}
      />
    );
  },
};

if (document.currentScript) {
  const apiUrl = document.currentScript.getAttribute('data-api-url');
  if (apiUrl) {
    window.IlariaChatbot.init({ apiUrl });
  }
}
