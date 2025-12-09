  import React, { useState, useEffect } from 'react';
import ChatButton from './ChatButton';
import ChatWindow from './ChatWindow';
import { chatApi } from '../services/api';

const ChatWidget = ({ apiUrl }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setMessages([{
      id: 1,
      role: 'assistant',
      content: 'გამარჯობა! მე ვარ ილარია HR სისტემის ასისტენტი. როგორ შემიძლია დაგეხმაროთ?',
      timestamp: new Date().toISOString()
    }]);
  }, []);

  const sendMessage = async (content) => {
    if (!content.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const history = [...messages, userMessage].slice(-10).map(m => ({
        role: m.role,
        content: m.content,
      }));

      const data = await chatApi.sendMessage(
        content.trim(),
        history,
        apiUrl,
      );

      const assistantMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Chat error:', error);

      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'უკაცრავად, დაფიქსირდა შეცდომა. გთხოვთ სცადოთ თავიდან.',
        timestamp: new Date().toISOString(),
        isError: true,
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="ilaria-chatbot-widget">
      {isOpen && (
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          onSend={sendMessage}
          onClose={() => setIsOpen(false)}
        />
      )}
      <ChatButton
        isOpen={isOpen}
        onClick={() => setIsOpen(!isOpen)}
      />
    </div>
  );
};

export default ChatWidget;
