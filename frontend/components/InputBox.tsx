import React, { useState, useRef, useEffect } from 'react';

interface InputBoxProps {
  onSend: (text: string) => void;
  disabled: boolean;
}

const InputBox: React.FC<InputBoxProps> = ({ onSend, disabled }) => {
  const [input, setInput] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 120) + 'px';
    }
  }, [input]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (input.trim() && !disabled) {
      onSend(input);
      setInput('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
        textareaRef.current.focus();
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="p-4 bg-white border-t border-gray-100">
      <form onSubmit={handleSubmit} className="relative flex items-end gap-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="დაწერეთ თქვენი კითხვა..."
          disabled={disabled}
          rows={1}
          className="flex-1 max-h-[120px] py-3 pl-4 pr-12 bg-gray-50 border-0 rounded-2xl text-sm focus:ring-2 focus:ring-primary/20 focus:bg-white transition-all resize-none outline-none disabled:bg-gray-100 disabled:text-gray-400 text-gray-800 placeholder-gray-400"
        />
        <button
          type="submit"
          disabled={disabled || !input.trim()}
          className="absolute right-2 bottom-2 p-1.5 rounded-xl bg-primary text-white hover:bg-primary/90 disabled:bg-gray-200 disabled:text-gray-400 transition-colors shadow-sm"
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="m22 2-7 20-4-9-9-4Z"/><path d="M22 2 11 13"/></svg>
        </button>
      </form>
      <div className="text-center mt-2">
        <p className="text-[10px] text-gray-400">ილარიას ასისტენტს შეუძლია შეცდომის დაშვება.</p>
      </div>
    </div>
  );
};

export default InputBox;