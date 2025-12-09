// src/modes/ChatMode.tsx
import React, { useState, useRef, useEffect } from "react";
import {
  sendChatMessage,
  ConversationMessage,
} from "../services/api";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
}

const ChatMode: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement | null>(null);

  // scroll always down
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: trimmed,
    };

    // ვამატებთ user მესიჯს UI-ში
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      // history backend-ის ფორმატით
      const historyForBackend: ConversationMessage[] = [
        ...messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
        {
          role: "user",
          content: userMessage.content,
        },
      ];

      // ვუგზავნით backend-ს
      const response = await sendChatMessage(
        userMessage.content,
        historyForBackend
      );

      const assistantMessage: Message = {
        id: crypto.randomUUID(),
        role: "assistant",
        content: response.answer,
      };

      setMessages((prev) => [...prev, assistantMessage]);

      // სურვილის შემთხვევაში, შეიძლება response.sources / metadata სადღაც ცალკე აჩვენო
      console.log("Sources:", response.sources);
      console.log("Metadata:", response.metadata);
    } catch (error) {
      console.error("❌ chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: crypto.randomUUID(),
          role: "assistant",
          content:
            "დაფიქსირდა შეცდომა backend-თან კავშირში. გადაამოწმე სერვერი და სცადე კიდევ ერთხელ.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown: React.KeyboardEventHandler<HTMLTextAreaElement> = (
    e
  ) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[800px] border rounded-xl bg-[#050509] text-white">
      {/* მესიჯები */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`max-w-[80%] rounded-2xl px-3 py-2 text-sm leading-relaxed ${
              m.role === "user"
                ? "ml-auto bg-[#3b82f6]"
                : "mr-auto bg-[#111827] border border-white/5"
            }`}
          >
            {m.content}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      {/* input */}
      <div className="border-t border-white/10 p-3">
        <div className="flex items-end gap-2">
          <textarea
            className="flex-1 resize-none rounded-xl bg-[#020617] border border-white/10 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/60"
            rows={2}
            placeholder="დაწერე კითხვა / აზრი..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isLoading}
          />
          <button
            className="shrink-0 rounded-xl px-4 py-2 text-sm font-medium bg-blue-600 hover:bg-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
          >
            გაგზავნა
          </button>
        </div>
        {isLoading && (
          <p className="mt-1 text-[11px] text-white/50">
            მოდელი ფიქრობს შენს პასუხზე…
          </p>
        )}
      </div>
    </div>
  );
};

export default ChatMode;
