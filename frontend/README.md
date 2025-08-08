# docs-to-eval Frontend

A modern, beautiful Next.js frontend for the docs-to-eval system, built with TypeScript, Tailwind CSS, and shadcn/ui.

## 🎉 BIG REFACTOR COMPLETE!

This frontend has been completely refactored from a plain HTML/CSS/JS application to a modern React/Next.js application with:

- ✅ **Next.js 15** with App Router
- ✅ **TypeScript** for type safety
- ✅ **Tailwind CSS** for styling
- ✅ **shadcn/ui** for beautiful, accessible components
- ✅ **React Hook Form** with Zod validation
- ✅ **Sonner** for beautiful toast notifications
- ✅ **Lucide React** for consistent icons
- ✅ **Responsive design** that works on all devices

## Features

### 🔬 Evaluation Tab
- **Corpus Input**: Text input or file/folder upload
- **Multiple Evaluation Types**: Auto-detect or manually select
- **Agentic Generation**: AI-powered question generation
- **Fine-tuning Support**: Automatic train/test splits for LoRA
- **Real-time Progress**: WebSocket updates with progress bars
- **Qwen Local Testing**: Test without API keys

### 📊 Classification Tab  
- **Smart Classification**: Automatically classify corpus types
- **Confidence Scoring**: See how confident the classification is
- **Multiple Input Methods**: Text or file uploads
- **Detailed Analysis**: Get reasoning behind classifications

### 📈 Runs Tab
- **Run Management**: View all evaluation runs
- **Fine-tuning Integration**: See LoRA training results
- **Performance Metrics**: Track improvements over time
- **Easy Downloads**: Export results with one click

### ⚙️ Settings Tab
- **Multiple Providers**: OpenRouter, OpenAI, Anthropic
- **API Key Management**: Secure storage and testing
- **Model Selection**: Choose from latest models
- **Environment Variables**: Support for server-side config

## Quick Start

### Prerequisites
- Node.js 18+ and npm
- Backend API server running (see main project README)

### Installation

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

The application will be available at `http://localhost:3000`

## Development

### Project Structure

```
src/
├── app/                    # Next.js App Router
│   ├── layout.tsx         # Root layout with providers
│   └── page.tsx           # Main page
├── components/
│   ├── ui/                # shadcn/ui components
│   ├── docs-to-eval-app.tsx    # Main app component
│   ├── evaluation-tab.tsx      # Evaluation functionality
│   ├── classification-tab.tsx  # Classification functionality
│   ├── runs-tab.tsx            # Runs management
│   └── settings-tab.tsx        # Settings configuration
└── lib/
    └── utils.ts           # Utility functions
```

### Key Technologies

- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe JavaScript
- **Tailwind CSS**: Utility-first CSS framework  
- **shadcn/ui**: High-quality React components
- **React Hook Form**: Performant forms with validation
- **Zod**: Schema validation
- **Sonner**: Beautiful toast notifications
- **Lucide React**: Icon library

### API Integration

The frontend communicates with the backend API at:
- **Base URL**: `/api/v1/`
- **WebSocket**: `/api/v1/ws/{run_id}` for real-time updates
- **File Upload**: Supports multiple files and folders
- **Authentication**: API key-based authentication

## Configuration

### Environment Variables

```bash
# Optional - can also be configured via UI
OPENROUTER_API_KEY=your-api-key-here
DOCS_TO_EVAL_MODEL_NAME=anthropic/claude-sonnet-4
DOCS_TO_EVAL_PROVIDER=openrouter
```

### Supported Models

- **GPT-5** (Latest & Best)
- **GPT-5-mini** (Fast & Cheap)
- **Claude Sonnet 4** (Anthropic)

## Deployment

### Production Build

```bash
npm run build
npm start
```

### Docker (if needed)

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```

## Features Highlight

### Beautiful UI
- **Modern Design**: Clean, professional interface
- **Responsive**: Works perfectly on desktop, tablet, and mobile
- **Dark Mode Ready**: Built with CSS variables for easy theming
- **Accessibility**: All components follow WCAG guidelines

### File Handling
- **Multiple File Types**: txt, md, py, js, json, csv, html, xml, yml, cfg, ini, log
- **Folder Upload**: Select entire directories
- **Progress Tracking**: See upload progress in real-time
- **Size Limits**: Automatic file size validation

### Real-time Updates
- **WebSocket Integration**: Live progress updates
- **Toast Notifications**: Beautiful, non-intrusive notifications
- **Progress Bars**: Visual feedback for long-running tasks
- **Error Handling**: Graceful error messages and recovery

### Form Validation
- **Schema-based**: Zod schemas for type-safe validation
- **Real-time**: Validate as you type
- **User-friendly**: Clear error messages
- **Accessibility**: Screen reader friendly

## Migration Notes

This frontend replaces the previous HTML-based interface with:

1. **Better Performance**: React's virtual DOM and Next.js optimizations
2. **Type Safety**: TypeScript catches errors at compile time
3. **Better UX**: Modern components and interactions
4. **Maintainability**: Modular component architecture
5. **Future-proof**: Built on modern React patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Support

- **Issues**: Report bugs via GitHub issues
- **Documentation**: See main project README
- **API Docs**: Check backend API documentation

---

**🚀 The frontend has been completely modernized with Next.js, TypeScript, and shadcn/ui for the best developer and user experience!**