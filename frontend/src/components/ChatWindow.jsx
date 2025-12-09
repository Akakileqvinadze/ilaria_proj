import React from 'react';
import { X } from 'lucide-react';
import MessageList from './MessageList';
import InputBox from './InputBox';

const ChatWindow = ({ messages, isLoading, onSend, onClose }) => {
  return (
    <div className="fixed bottom-24 right-6 w-96 h-[600px] bg-white rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50">
      <div className="bg-gradient-to-r from-primary to-secondary text-white p-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-white/20 rounded-full flex items-center justify-center">
            <span className="text-2xl">🤖</span>
          </div>
          <div>
            <h3 className="font-semibold">ილარია ასისტენტი</h3>
            <p className="text-xs text-white/80">ონლაინ</p>
          </div>
        </div>
        <button onClick={onClose} className="p-1.5 hover:bg-white/20 rounded-lg">
          <X className="w-5 h-5" />
        </button>
      </div>

      <MessageList messages={messages} isLoading={isLoading} />
      <InputBox onSend={onSend} disabled={isLoading} />
    </div>
  );
};

export default ChatWindow;
