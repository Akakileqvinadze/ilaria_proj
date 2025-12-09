import React from 'react';
import { Message as MessageType } from '../types';

interface MessageProps {
  message: MessageType;
}

// Simple internal Markdown renderer to avoid heavy dependencies in this environment
const SimpleMarkdown: React.FC<{ content: string }> = ({ content }) => {
  // Handle newlines
  const paragraphs = content.split('\n');
  
  return (
    <div className="prose prose-sm max-w-none text-inherit">
      {paragraphs.map((line, idx) => {
        if (!line.trim()) return <div key={idx} className="h-2"></div>;
        
        // Simple bold parser for **text**
        const parts = line.split(/(\*\*.*?\*\*)/g);
        
        return (
          <p key={idx} className="mb-1 last:mb-0">
            {parts.map((part, pIdx) => {
              if (part.startsWith('**') && part.endsWith('**')) {
                return <strong key={pIdx}>{part.slice(2, -2)}</strong>;
              }
              return part;
            })}
          </p>
        );
      })}
    </div>
  );
};

const Message: React.FC<MessageProps> = ({ message }) => {
  const isUser = message.role === 'user';

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
      <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-sm ${
        isUser ? 'bg-primary' : 'bg-gray-200'
      }`}>
        {isUser ? (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-gray-600"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>
        )}
      </div>

      <div className={`flex-1 max-w-[85%] ${isUser ? 'text-right' : 'text-left'}`}>
        <div className={`inline-block p-3.5 rounded-2xl text-sm leading-relaxed shadow-sm ${
          isUser
            ? 'bg-primary text-white rounded-tr-none'
            : message.isError 
              ? 'bg-red-50 text-red-800 border border-red-200 rounded-tl-none'
              : 'bg-white text-gray-800 border border-gray-100 rounded-tl-none'
        }`}>
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <SimpleMarkdown content={message.content} />
          )}
        </div>
        
        <p className={`text-[10px] mt-1 px-2 ${isUser ? 'text-gray-400' : 'text-gray-400'}`}>
          {new Date(message.timestamp).toLocaleTimeString('ka-GE', {
            hour: '2-digit',
            minute: '2-digit',
          })}
        </p>
      </div>
    </div>
  );
};

export default Message;