import axios from 'axios';

const AI_SERVICE_URL = process.env.AI_SERVICE_URL || 'http://localhost:8000';

export const chatController = {
  async sendMessage(req, res, next) {
    try {
      const { message, conversationHistory = [] } = req.body;

      if (!message?.trim()) {
        return res.status(400).json({
          success: false,
          error: 'Message is required'
        });
      }

      const limitedHistory = conversationHistory.slice(-10);

      const startTime = Date.now();
      const aiResponse = await axios.post(
        `${AI_SERVICE_URL}/api/chat`,
        {
          message: message.trim(),
          conversation_history: limitedHistory
        },
        { timeout: 30000 }
      );
      const responseTime = Date.now() - startTime;

      console.log(`✅ Response in ${responseTime}ms`);

      res.json({
        success: true,
        data: {
          answer: aiResponse.data.answer,
          sources: aiResponse.data.sources || [],
          metadata: {
            ...aiResponse.data.metadata,
            response_time_ms: responseTime
          },
          timestamp: new Date().toISOString()
        }
      });

    } catch (error) {
      console.error('❌ Chat error:', error.message);
      
      if (error.code === 'ECONNREFUSED') {
        return res.status(503).json({
          success: false,
          error: 'AI service unavailable'
        });
      }

      next(error);
    }
  },

  async healthCheck(req, res) {
    try {
      const aiHealth = await axios.get(`${AI_SERVICE_URL}/health`, {
        timeout: 5000
      });

      res.json({
        success: true,
        backend: 'healthy',
        ai_service: aiHealth.data
      });
    } catch (error) {
      res.status(503).json({
        success: false,
        backend: 'healthy',
        ai_service: 'unavailable'
      });
    }
  }
};