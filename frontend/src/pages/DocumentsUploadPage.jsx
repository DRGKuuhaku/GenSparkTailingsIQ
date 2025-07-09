import React, { useState } from 'react';
import { Box, Typography, Button, Paper, LinearProgress, Alert } from '@mui/material';
import { tailingsIQApi } from '../api/tailingsIQApi';

const DocumentsUploadPage = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [success, setSuccess] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setSuccess(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setSuccess(null);
    setError(null);
    const formData = new FormData(); 
    formData.append('file', file);
    try {
      await tailingsIQApi.post('/documents/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setSuccess('File uploaded successfully!');
      setFile(null);
    } catch (err) {
      setError('Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <Box sx={{ maxWidth: 500, mx: 'auto', mt: 6, p: 2 }}>
      <Typography variant="h4" sx={{ mb: 2, fontWeight: 700, textAlign: 'center' }}>
        Upload Document
      </Typography>
      <Paper elevation={3} sx={{ p: 3, display: 'flex', flexDirection: 'column', gap: 2 }}>
        <input
          type="file"
          onChange={handleFileChange}
          disabled={uploading}
          style={{ marginBottom: 16 }}
        />
        <Button
          variant="contained"
          color="primary"
          onClick={handleUpload}
          disabled={!file || uploading}
          sx={{ backgroundColor: '#006039', '&:hover': { backgroundColor: '#004d2e' } }}
        >
          {uploading ? 'Uploading...' : 'Upload'}
        </Button>
        {uploading && <LinearProgress sx={{ mt: 1 }} />}
        {success && <Alert severity="success">{success}</Alert>}
        {error && <Alert severity="error">{error}</Alert>}
      </Paper>
    </Box>
  );
};

export default DocumentsUploadPage; 