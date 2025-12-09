// src/services/api.ts
export type ConversationRole = "user" | "assistant";

export interface ConversationMessage {
  role: ConversationRole;
  content: string;
}

export interface Source {
  id: number;
  content: string;
  section: string;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  metadata: Record<string, any>;
}

const API_URL =
  import.meta.env.VITE_BACKEND_URL || "http://localhost:5001";

/**
 * აგზავნის ჩატის შეტყობინებას backend-ზე /api/chat ენდპოინტზე
 */
export async function sendChatMessage(
  message: string,
  conversation_history: ConversationMessage[]
): Promise<ChatResponse> {
  const res = await fetch(`${API_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      conversation_history,
    }),
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    console.error("❌ Backend error:", res.status, text);
    throw new Error(
      text || `Backend error: ${res.status} ${res.statusText}`
    );
  }

  const data = (await res.json()) as ChatResponse;
  return data;
}
