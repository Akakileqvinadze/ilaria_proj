import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import rateLimit from 'express-rate-limit';
import chatRoutes from './routes/chat.routes.js';
import { errorHandler } from './middleware/errorHandler.js';

dotenv.config();

const app = express();
const PORT = process.env.PORT || 5001;

// Rate limiting
const limiter = rateLimit({
  windowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS) || 60000,
  max: parseInt(process.env.RATE_LIMIT_MAX_REQUESTS) || 30,
  message: { success: false, error: 'Too many requests' }
});

// Middleware
app.use(express.json({ limit: '10kb' }));
app.use(cors({
  origin: process.env.WIDGET_ORIGIN || '*',
  credentials: true
}));
app.use('/api/', limiter);

// Routes
app.use('/api', chatRoutes);

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', service: 'ilaria-backend' });
});

// Error handling
app.use(errorHandler);

app.listen(PORT, () => {
  console.log(`✅ Backend running on port ${PORT}`);
  console.log(`🔗 AI Service: ${process.env.AI_SERVICE_URL}`);
});