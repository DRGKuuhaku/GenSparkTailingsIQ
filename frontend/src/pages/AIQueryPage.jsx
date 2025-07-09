import React, { useState } from 'react';
import { Box, Typography, TextField, Button, Paper, CircularProgress, List, ListItem, ListItemText, Avatar } from '@mui/material';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import { tailingsIQApi } from '../api/tailingsIQApi';

const AIQueryPage = () => {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hi! I am TailingsIQ AI. How can I help you today?' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  // Helper to convert messages to OpenAI format
  const getOpenAIMessages = () =>
    messages.map(msg => ({
      role: msg.sender === 'ai' ? 'assistant' : 'user',
      content: msg.text
    }));

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMessage = { sender: 'user', text: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Send full chat history to backend
      const response = await tailingsIQApi.post('/ai-query', {
        messages: [
          ...getOpenAIMessages(),
          { role: 'user', content: input }
        ]
      });
      setMessages((prev) => [
        ...prev,
        { sender: 'ai', text: response.data.answer }
      ]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: 'ai', text: 'Sorry, there was an error contacting the AI service.' }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter' && !loading) {
      handleSend();
    }
  };

  return (
    <Box sx={{ maxWidth: 800, mx: 'auto', mt: 6, p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2, fontWeight: 700, textAlign: 'center' }}>
        AI Query Chat
      </Typography>
      <Paper elevation={3} sx={{ p: 2, minHeight: 600, display: 'flex', flexDirection: 'column', mb: 2 }}>
        <List sx={{ flexGrow: 1, overflowY: 'auto', mb: 1 }}>
          {messages.map((msg, idx) => (
            <ListItem key={idx} alignItems={msg.sender === 'user' ? 'right' : 'left'} sx={{ justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start' }}>
              {msg.sender === 'ai' && <Avatar sx={{ bgcolor: '#006039', mr: 1 }}><SmartToyIcon /></Avatar>}
              <ListItemText
                primary={msg.text}
                sx={{
                  textAlign: msg.sender === 'user' ? 'right' : 'left',
                  background: msg.sender === 'user' ? '#e0f7fa' : '#f1f8e9',
                  borderRadius: 2,
                  px: 2,
                  py: 1,
                  maxWidth: 400,
                  display: 'inline-block',
                }}
              />
            </ListItem>
          ))}
          {loading && (
            <ListItem>
              <CircularProgress size={24} sx={{ color: '#006039' }} />
              <ListItemText primary="AI is typing..." sx={{ ml: 2, color: '#888' }} />
            </ListItem>
          )}
        </List>
        <Box sx={{ display: 'flex', gap: 1, mt: 1 }}>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your question..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleInputKeyDown}
            disabled={loading}
          />
          <Button
            variant="contained"
            color="primary"
            onClick={handleSend}
            disabled={loading || !input.trim()}
            sx={{ minWidth: 100, backgroundColor: '#006039', '&:hover': { backgroundColor: '#004d2e' } }}
          >
            Send
          </Button>
        </Box>
      </Paper>
      <Typography variant="body2" color="text.secondary" sx={{ textAlign: 'center' }}>
        This is a demo chat UI. AI answers will appear here when backend integration is complete.
      </Typography>
    </Box>
  );
};

export default AIQueryPage;
