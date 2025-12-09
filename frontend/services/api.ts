import { API_URL, DEMO_FALLBACK_RESPONSE } from '../constants';
import { ChatResponse, Message } from '../types';

export const chatApi = {
  sendMessage: async (message: string, conversationHistory: Message[]): Promise<ChatResponse> => {
    try {
      const historyPayload = conversationHistory.map(m => ({
        role: m.role,
        content: m.content
      }));

      // 🔴 /chat → ✅ /api/chat
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message,
          conversationHistory: historyPayload
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('API Call failed:', error);

      await new Promise(resolve => setTimeout(resolve, 1500));

      return {
        success: true,
        data: {
          answer: DEMO_FALLBACK_RESPONSE,
          sources: [],
          metadata: {
            model: 'demo-fallback',
            retrieved_docs: 0
          },
          timestamp: new Date().toISOString()
        }
      };
    }
  },

  healthCheck: async () => {
    try {
      // ესეც დიდი შანსია რომ /api/health იყოს
      const response = await fetch(`${API_URL}/api/health`);
      return await response.json();
    } catch (error) {
      return { success: false, error: 'Service unavailable' };
    }
  }
};
