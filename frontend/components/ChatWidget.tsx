import React, { useState, useEffect } from 'react';
import ChatButton from './ChatButton';
import ChatWindow from './ChatWindow';
import { chatApi } from '../services/api';
import { Message } from '../types';
import { WELCOME_MESSAGE, INITIAL_MESSAGE_ID } from '../constants';

const ChatWidget: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Initialize with welcome message
  useEffect(() => {
    setMessages([{
      id: INITIAL_MESSAGE_ID,
      role: 'assistant',
      content: WELCOME_MESSAGE,
      timestamp: new Date().toISOString()
    }]);
  }, []);

  const sendMessage = async (content: string) => {
    if (!content.trim() || isLoading) return;

    // Add user message
    const userMessage: Message = {
      id: Date.now(),
      role: 'user',
      content: content.trim(),
      timestamp: new Date().toISOString()
    };
    
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);

    try {
      const history = messages.slice(-10); // Keep last 10 messages for context
      
      const response = await chatApi.sendMessage(content.trim(), history);

      if (response.success) {
        const assistantMessage: Message = {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.data.answer,
          timestamp: response.data.timestamp
        };
        setMessages(prev => [...prev, assistantMessage]);
      } else {
        throw new Error('Response unsuccessful');
      }

    } catch (error) {
      console.error('Chat error:', error);
      
      const errorMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'უკაცრავად, დაფიქსირდა ტექნიკური ხარვეზი. გთხოვთ სცადოთ მოგვიანებით.',
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="font-sans antialiased">
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