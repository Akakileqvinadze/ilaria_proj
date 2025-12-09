// src/modes/LiveMode.tsx
import React, { useEffect, useRef, useState } from "react";
// ისევ, ai იმპორტი შენი კონფიგურაციიდან
import { ai } from "../lib/aiClient";

type Role = "user" | "assistant";

interface Message {
  id: string;
  role: Role;
  content: string;
}

const LiveMode: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  const mediaStreamRef = useRef<MediaStream | null>(null);
  const liveSessionRef = useRef<any>(null);

  const modelName =
    import.meta.env.VITE_GEMINI_LIVE_MODEL || "models/gemini-2.0-flash";

  useEffect(() => {
    let cancelled = false;

    const connect = async () => {
      try {
        const live = await ai.live.connect({
          model: modelName,

          // ❗ აქ იყო შენი შეცდომა – აქ აღარ უნდა იყოს model ველი
          inputAudioTranscription: {}, // ასე აქტიურდება შემომავალი აუდიოს ტრანსკრიფცია
          outputAudioTranscription: {}, // ასე აქტიურდება გამავალი აუდიოს ტრანსკრიფცია
        });

        if (cancelled) return;

        liveSessionRef.current = live;
        setIsConnected(true);

        // მოვუსმინოთ content/text event-ებს
        live.on("content", (event: any) => {
          const parts: string[] = [];

          for (const part of event.parts ?? []) {
            if (part.text) parts.push(part.text);
          }

          const text = parts.join(" ");
          if (!text) return;

          const role: Role =
            event.role === "user" || event.role === "USER"
              ? "user"
              : "assistant";

          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              role,
              content: text,
            },
          ]);
        });

        live.on("error", (err: any) => {
          console.error("❌ Live error:", err);
        });
      } catch (error) {
        console.error("❌ Failed to connect live session:", error);
      }
    };

    void connect();

    return () => {
      cancelled = true;
      try {
        liveSessionRef.current?.close?.();
      } catch {
        // ignore
      }
    };
  }, [modelName]);

  const startRecording = async () => {
    if (isRecording || !isConnected) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      setIsRecording(true);

      // აქ შეგიძლია დაამატო MediaRecorder / AudioWorklet ლოგიკა,
      // რომ აუდიო chunk-ებად გაუშვა liveSessionRef.current-ში.
      // ახლა skeleton-ად გიტოვებ.
    } catch (error) {
      console.error("❌ Mic access error:", error);
    }
  };

  const stopRecording = () => {
    setIsRecording(false);
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
    // თუ Recorder გაქვს, აქაც გააჩერე
  };

  return (
    <div className="flex flex-col h-full max-h-[800px] border rounded-xl bg-[#050509] text-white">
      <div className="p-3 border-b border-white/10 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm">
          <div
            className={`h-2 w-2 rounded-full ${
              isConnected ? "bg-green-400" : "bg-red-500"
            }`}
          />
          <span>{isConnected ? "Live session ჩართულია" : "კავშირი არ არის"}</span>
        </div>
        <button
          className={`px-3 py-1 rounded-full text-xs font-medium ${
            isRecording
              ? "bg-red-600 hover:bg-red-500"
              : "bg-emerald-600 hover:bg-emerald-500"
          } disabled:opacity-50 disabled:cursor-not-allowed`}
          onClick={isRecording ? stopRecording : startRecording}
          disabled={!isConnected}
        >
          {isRecording ? "ჩაწერის გაჩერება" : "ხმოვანი ჩაწერა"}
        </button>
      </div>

      {/* ტექსტური ლოგი */}
      <div className="flex-1 overflow-y-auto p-4 space-y-3 text-sm">
        {messages.map((m) => (
          <div
            key={m.id}
            className={`max-w-[80%] rounded-2xl px-3 py-2 leading-relaxed ${
              m.role === "user"
                ? "ml-auto bg-[#3b82f6]"
                : "mr-auto bg-[#111827] border border-white/5"
            }`}
          >
            {m.content}
          </div>
        ))}
      </div>
    </div>
  );
};

export default LiveMode;
