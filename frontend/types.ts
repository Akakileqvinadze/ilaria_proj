export interface Message {
  id: number;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  isError?: boolean;
}

export interface Source {
  id: number;
  content: string;
  section: string;
}

export interface ChatResponse {
  success: boolean;
  data: {
    answer: string;
    sources: Source[];
    metadata: {
      model: string;
      retrieved_docs: number;
      response_time_ms?: number;
    };
    timestamp: string;
  };
}

export interface ApiError {
  success: false;
  error: string;
}