import axios from 'axios';

const BASE_URL =
  (import.meta.env.VITE_API_URL && import.meta.env.VITE_API_URL.replace(/\/+$/, "")) ||
  'http://localhost:5000/api';

const apiClient = axios.create({
  baseURL: BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 30000,
});

export const chatApi = {
  sendMessage: async (message, conversationHistory, apiUrlOverride) => {
    const client = apiUrlOverride
      ? axios.create({
          baseURL: apiUrlOverride.replace(/\/+$/, ""),
          headers: { 'Content-Type': 'application/json' },
          timeout: 30000,
        })
      : apiClient;

    const payload = {
      message,
      conversationHistory,
      conversation_history: conversationHistory,
    };

    const response = await client.post('/chat', payload);
    return response.data;
  },

  healthCheck: async (apiUrlOverride) => {
    const client = apiUrlOverride
      ? axios.create({
          baseURL: apiUrlOverride.replace(/\/+$/, ""),
          timeout: 10000,
        })
      : apiClient;

    const response = await client.get('/health');
    return response.data;
  },
};
