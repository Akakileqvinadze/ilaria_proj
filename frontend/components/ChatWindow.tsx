import React from 'react';
import MessageList from './MessageList';
import InputBox from './InputBox';
import { Message } from '../types';

interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  onSend: (content: string) => void;
  onClose: () => void;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isLoading, onSend, onClose }) => {
  return (
    <div className="fixed bottom-24 right-6 w-[380px] h-[600px] max-h-[80vh] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50 border border-gray-100 animate-[fade-in_0.2s_ease-out]">
      {/* Header */}
      <div className="bg-gradient-to-r from-primary to-secondary p-4 flex items-center justify-between shadow-md z-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 backdrop-blur-sm rounded-full flex items-center justify-center border border-white/30">
            <span className="text-2xl">🤖</span>
          </div>
          <div>
            <h3 className="font-bold text-white text-base">ილარია ასისტენტი</h3>
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
              <p className="text-xs text-white/90 font-medium">ონლაინ</p>
            </div>
          </div>
        </div>
        <button 
          onClick={onClose} 
          className="p-2 hover:bg-white/20 rounded-xl transition-colors text-white/90 hover:text-white"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>
        </button>
      </div>

      {/* Messages */}
      <MessageList messages={messages} isLoading={isLoading} />

      {/* Input */}
      <InputBox onSend={onSend} disabled={isLoading} />
    </div>
  );
};

export default ChatWindow;