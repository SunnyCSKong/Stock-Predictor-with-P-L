import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  server: {
    proxy: {
      '/listings':'http://localhost:5000',
      '/submit_form': 'http://localhost:5000',
      '/get_plot': {target: 'http://localhost:5000', changeOrigin:true}
    },
  },
  plugins: [react()],
});

