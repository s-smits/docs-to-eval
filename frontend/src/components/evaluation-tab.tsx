"use client";

import { useState, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { FileText, Upload, FolderOpen, Trash2, Play, Zap, Settings } from "lucide-react";
import { toast } from "sonner";

const evaluationSchema = z.object({
  corpusText: z.string().optional(),
  evalType: z.string().optional(),
  numQuestions: z.number().min(1).max(200),
  temperature: z.number().min(0).max(2),
  maxConcurrent: z.number().min(1).max(20),
  tokenThreshold: z.number().min(500).max(4000),
  useAgentic: z.boolean(),
  finetuneEnabled: z.boolean(),
  finetunePercentage: z.number(),
  finetuneSeed: z.number(),
  runName: z.string().optional(),
});

type EvaluationFormData = z.infer<typeof evaluationSchema>;

interface FileWithPath {
  file: File;
  path: string;
}

export function EvaluationTab() {
  const [selectedFiles, setSelectedFiles] = useState<FileWithPath[]>([]);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [results, setResults] = useState<any>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [evaluationCompleted, setEvaluationCompleted] = useState(false);
  const [hasFineTuneData, setHasFineTuneData] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const form = useForm<EvaluationFormData>({
    resolver: zodResolver(evaluationSchema),
    defaultValues: {
      corpusText: "",
      evalType: "",
      runName: "",
      numQuestions: 50,
      temperature: 0.7,
      maxConcurrent: 5,
      tokenThreshold: 2000,
      useAgentic: true,
      finetuneEnabled: true,
      finetunePercentage: 0.2,
      finetuneSeed: 42,
    },
  });

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newFiles: FileWithPath[] = Array.from(files).map(file => ({
      file,
      path: file.name
    }));

    setSelectedFiles(newFiles);

    // If only one file, read and display content
    if (files.length === 1) {
      const reader = new FileReader();
      reader.onload = (e) => {
        form.setValue("corpusText", e.target?.result as string);
      };
      reader.readAsText(files[0]);
    } else if (files.length > 1) {
      // Clear textarea for multiple files
      form.setValue("corpusText", "");
    }

    toast.success(`Selected ${files.length} file${files.length > 1 ? 's' : ''}`);
  };

  const handleFolderSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files) return;

    const newFiles: FileWithPath[] = Array.from(files).map(file => ({
      file,
      path: file.webkitRelativePath || file.name
    }));

    setSelectedFiles(newFiles);

    // Clear textarea for multiple files
    if (files.length > 1) {
      form.setValue("corpusText", "");
    }

    toast.success(`Selected ${files.length} file${files.length > 1 ? 's' : ''} from folder`);
  };

  const clearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
    toast.info("Files cleared");
  };

  const loadSampleCorpus = () => {
    const sampleText = `# Machine Learning Fundamentals

## Neural Networks
Neural networks are computational models inspired by biological neural networks. A neural network consists of layers of interconnected nodes (neurons). Each connection has an associated weight that gets adjusted during training.

### Key Components:
1. **Input Layer**: Receives the initial data
2. **Hidden Layers**: Process the data through weighted connections and activation functions
3. **Output Layer**: Produces the final prediction or classification

### Training Process:
Neural networks learn through backpropagation, which calculates gradients of the loss function with respect to the weights. The learning rate determines how much the weights are adjusted in each iteration.

Common activation functions include:
- ReLU (Rectified Linear Unit): f(x) = max(0, x)
- Sigmoid: f(x) = 1 / (1 + e^(-x))
- Tanh: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

## Optimization Algorithms
Gradient Descent is the most fundamental optimization algorithm. Variations include:
- Stochastic Gradient Descent (SGD)
- Adam Optimizer
- RMSprop
- AdaGrad

## Overfitting and Regularization
Overfitting occurs when a model learns the training data too well, including noise. Regularization techniques help prevent this:
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- Dropout
- Early Stopping

## Performance Metrics
For classification problems:
- Accuracy = (TP + TN) / (TP + TN + FP + FN)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1-Score = 2 * (Precision * Recall) / (Precision + Recall)

For regression problems:
- Mean Squared Error (MSE) = Σ(y_true - y_pred)² / n
- Mean Absolute Error (MAE) = Σ|y_true - y_pred| / n
- R-squared = 1 - (SS_res / SS_tot)`;

    form.setValue("corpusText", sampleText);
    form.setValue("evalType", "domain_knowledge");
    toast.success("Sample corpus loaded");
  };

  const onSubmit = async (data: EvaluationFormData) => {
    // Check if we have either text or files
    if (!data.corpusText?.trim() && selectedFiles.length === 0) {
      toast.error("Please enter corpus text or select files/folder to evaluate");
      return;
    }

    setIsEvaluating(true);
    setProgress(0);
    setStatusMessage("Starting evaluation...");
    setResults(null);
    setEvaluationCompleted(false);
    setHasFineTuneData(false);

    try {
      let corpusText = data.corpusText;

      // Handle multiple files vs text input
      if (selectedFiles.length > 0) {
        // Upload multiple files
        const formData = new FormData();
        selectedFiles.forEach(({ file }) => {
          formData.append('files', file);
        });

        if (data.runName) {
          formData.append('name', data.runName);
        }

        // Upload files first
        const uploadResponse = await fetch('/api/v1/corpus/upload-multiple', {
          method: 'POST',
          body: formData
        });

        if (!uploadResponse.ok) {
          throw new Error(`File upload failed: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        corpusText = uploadResult.corpus_text;

        toast.success(`Uploaded ${uploadResult.files_processed} files successfully`);
      }

      // Start evaluation with corpus text
      const evaluationData = {
        corpus_text: corpusText,
        eval_type: data.evalType || null,
        num_questions: data.numQuestions,
        use_agentic: data.useAgentic,
        temperature: data.temperature,
        max_concurrent: data.maxConcurrent,
        token_threshold: data.tokenThreshold,
        run_name: data.runName || null,
        finetune_test_set_enabled: data.finetuneEnabled,
        finetune_test_set_percentage: data.finetunePercentage,
        finetune_random_seed: data.finetuneSeed
      };

      const response = await fetch('/api/v1/evaluation/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(evaluationData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setCurrentRunId(result.run_id);

      // Connect to WebSocket for real-time updates
      connectWebSocket(result.run_id);

      // Poll for completion
      pollEvaluationStatus(result.run_id);

    } catch (error: any) {
      console.error('Evaluation error:', error);

      // Check if it's an API key error
      if (error.message.includes('API key is required') || error.message.includes('400')) {
        toast.error('API Key Required: Please set your OpenRouter API key in Settings tab before running agentic evaluation.');
      } else {
        toast.error(`Evaluation failed: ${error.message}`);
      }
      setIsEvaluating(false);
    }
  };

  const connectWebSocket = (runId: string) => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/${runId}`;

    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'progress') {
        setProgress(data.progress);
        setStatusMessage(data.message);
      } else if (data.type === 'status') {
        toast.info(data.message);
      } else if (data.type === 'complete') {
        handleEvaluationComplete(runId);
      } else if (data.type === 'error') {
        toast.error(data.message);
        setIsEvaluating(false);
      }
    };

    ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  };

  const pollEvaluationStatus = async (runId: string) => {
    const maxAttempts = 120; // 10 minutes
    let attempts = 0;

    const poll = async () => {
      try {
        const response = await fetch(`/api/v1/evaluation/${runId}/status`);

        if (response.status === 404) {
          toast.error('Evaluation session was lost. Please start a new evaluation.');
          setIsEvaluating(false);
          return;
        }

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const status = await response.json();

        if (status.status === 'completed') {
          handleEvaluationComplete(runId);
          return;
        } else if (status.status === 'failed') {
          toast.error(status.error || 'Evaluation failed');
          setIsEvaluating(false);
          return;
        }

        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000); // Poll every 5 seconds
        } else {
          toast.error('Evaluation timeout');
          setIsEvaluating(false);
        }
      } catch (error: any) {
        console.error('Status polling error:', error);
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000);
        } else {
          toast.error('Failed to get evaluation status after multiple attempts');
          setIsEvaluating(false);
        }
      }
    };

    setTimeout(poll, 2000); // Start polling after 2 seconds
  };

  const handleEvaluationComplete = async (runId: string) => {
    try {
      const response = await fetch(`/api/v1/evaluation/${runId}/results`);
      const resultWrapper = await response.json();

      // Extract the actual results
      const results = resultWrapper.results || resultWrapper;

      setResults(results);
      setIsEvaluating(false);
      setEvaluationCompleted(true);
      
      // Check if fine-tuning data is available
      const hasFinetune = results?.finetune_test_set?.enabled || false;
      setHasFineTuneData(hasFinetune);
      
      toast.success("Evaluation completed successfully!");

    } catch (error: any) {
      console.error('Results fetch error:', error);
      toast.error('Failed to fetch results');
      setIsEvaluating(false);
    }
  };

  const startQwenEvaluation = async () => {
    const corpusText = form.getValues("corpusText");

    if (!corpusText?.trim() && selectedFiles.length === 0) {
      toast.error('Please enter corpus text or select files for local testing');
      return;
    }

    setIsEvaluating(true);
    setProgress(0);
    setStatusMessage("Starting quick local test...");
    setResults(null);
    setEvaluationCompleted(false);
    setHasFineTuneData(false);

    try {
      let finalCorpusText = corpusText;

      if (selectedFiles.length > 0) {
        // Upload multiple files first
        const formData = new FormData();
        selectedFiles.forEach(({ file }) => {
          formData.append('files', file);
        });

        const runName = form.getValues("runName") || 'Qwen Local Test';
        formData.append('name', runName);

        const uploadResponse = await fetch('/api/v1/corpus/upload-multiple', {
          method: 'POST',
          body: formData
        });

        if (!uploadResponse.ok) {
          throw new Error(`File upload failed: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        finalCorpusText = uploadResult.corpus_text;

        toast.success(`Uploaded ${uploadResult.files_processed} files for local testing`);
      }

      const qwenData = {
        corpus_text: finalCorpusText,
        num_questions: form.getValues("numQuestions") || 5,
        use_fictional: true,
        token_threshold: form.getValues("tokenThreshold") || 2000,
        run_name: form.getValues("runName") || 'Qwen Local Test'
      };

      const response = await fetch('/api/v1/evaluation/qwen-local', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(qwenData)
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setCurrentRunId(result.run_id);

      toast.info('🚀 Quick local test started - no API key required!');

      connectWebSocket(result.run_id);
      pollEvaluationStatus(result.run_id);

    } catch (error: any) {
      console.error('Qwen evaluation error:', error);
      toast.error(`Qwen evaluation failed: ${error.message}`);
      setIsEvaluating(false);
    }
  };

  const downloadResults = () => {
    if (currentRunId) {
      window.open(`/api/v1/evaluation/${currentRunId}/download`, '_blank');
    }
  };

  const openFinetuningDashboard = (runId: string) => {
    const url = `/api/v1/evaluation/${runId}/lora-finetune/dashboard?run_id=${runId}`;
    window.open(url, '_blank');
  };

  const totalSize = selectedFiles.reduce((acc, { file }) => acc + file.size, 0);

  return (
    <div className="space-y-6">
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          {/* Corpus Input */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5" />
                Corpus Input
              </CardTitle>
              <CardDescription>
                Enter your corpus text directly or upload files/folders
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex gap-6">
                {/* Left side - Upload Files */}
                <div className="flex-1 space-y-3">
                  <Label>Upload Files or Folder</Label>
                  <div className="flex gap-3 flex-wrap">
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileSelect}
                      multiple
                      accept=".txt,.md,.py,.js,.json,.csv,.html,.xml,.yml,.yaml,.cfg,.ini,.log"
                      className="hidden"
                    />
                    <input
                      type="file"
                      ref={folderInputRef}
                      onChange={handleFolderSelect}
                      {...({ webkitdirectory: "" } as any)}
                      multiple
                      className="hidden"
                    />
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => fileInputRef.current?.click()}
                      className="flex items-center gap-2"
                    >
                      <Upload className="h-4 w-4" />
                      Select Files
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={() => folderInputRef.current?.click()}
                      className="flex items-center gap-2"
                    >
                      <FolderOpen className="h-4 w-4" />
                      Select Folder
                    </Button>
                    <Button
                      type="button"
                      variant="outline"
                      onClick={loadSampleCorpus}
                    >
                      Load Sample
                    </Button>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    Supported formats: txt, md, py, js, json, csv, html, xml, yml, cfg, ini, log
                  </p>

                  {selectedFiles.length > 0 && (
                    <Card className="p-4 bg-muted/50">
                      <div className="flex items-center justify-between mb-3">
                        <h4 className="font-medium">Selected Files</h4>
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          onClick={clearFiles}
                          className="text-destructive hover:text-destructive"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="space-y-1 max-h-24 overflow-y-auto">
                        {selectedFiles.map(({ file, path }, index) => (
                          <div key={index} className="flex items-center justify-between text-sm">
                            <span className="truncate">{path}</span>
                            <span className="text-muted-foreground">{formatFileSize(file.size)}</span>
                          </div>
                        ))}
                      </div>
                      <div className="flex items-center justify-between mt-3 pt-3 border-t text-sm">
                        <span className="font-medium">{selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''} selected</span>
                        <span className="font-medium">{formatFileSize(totalSize)} total</span>
                      </div>
                    </Card>
                  )}
                </div>

                {/* Right side - Corpus Text */}
                <div className="flex-1">
                  <FormField
                    control={form.control}
                    name="corpusText"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Corpus Text (Optional)</FormLabel>
                        <FormControl>
                          <Textarea
                            {...field}
                            placeholder="Enter your corpus text here or use the file upload on the left..."
                            className="min-h-[240px]"
                          />
                        </FormControl>
                        <FormMessage />
                      </FormItem>
                    )}
                  />
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Evaluation Settings */}
          <Card>
            <CardHeader>
              <CardTitle>Evaluation Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="evalType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Evaluation Type</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Auto-detect" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="auto-detect">Auto-detect</SelectItem>
                          <SelectItem value="mathematical">Mathematical</SelectItem>
                          <SelectItem value="code_generation">Code Generation</SelectItem>
                          <SelectItem value="factual_qa">Factual Q&A</SelectItem>
                          <SelectItem value="multiple_choice">Multiple Choice</SelectItem>
                          <SelectItem value="summarization">Summarization</SelectItem>
                          <SelectItem value="translation">Translation</SelectItem>
                          <SelectItem value="creative_writing">Creative Writing</SelectItem>
                          <SelectItem value="commonsense_reasoning">Commonsense Reasoning</SelectItem>
                          <SelectItem value="reading_comprehension">Reading Comprehension</SelectItem>
                          <SelectItem value="domain_knowledge">Domain Knowledge</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="tokenThreshold"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Token Threshold ({field.value} tokens)</FormLabel>
                      <FormControl>
                        <Slider
                          min={500}
                          max={4000}
                          step={100}
                          value={[field.value]}
                          onValueChange={(value) => field.onChange(value[0])}
                          className="w-full"
                        />
                      </FormControl>
                      <FormDescription>
                        Minimum token threshold for chunk concatenation (500-4000 tokens)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="numQuestions"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Number of Questions</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={1}
                          max={200}
                          {...field}
                          onChange={(e) => {
                            const value = parseInt(e.target.value);
                            field.onChange(isNaN(value) ? undefined : value);
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="temperature"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Temperature</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={0}
                          max={2}
                          step={0.1}
                          {...field}
                          onChange={(e) => {
                            const value = parseFloat(e.target.value);
                            field.onChange(isNaN(value) ? undefined : value);
                          }}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="maxConcurrent"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Concurrent Requests</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={1}
                          max={20}
                          {...field}
                          onChange={(e) => {
                            const value = parseInt(e.target.value);
                            field.onChange(isNaN(value) ? undefined : value);
                          }}
                        />
                      </FormControl>
                      <FormDescription>
                        Number of parallel API requests (1-20). Higher values are faster but may hit rate limits.
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <FormField
                  control={form.control}
                  name="useAgentic"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">Use Agentic Generation</FormLabel>
                        <FormDescription>
                          Enable AI-powered contextual question generation
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="finetuneEnabled"
                  render={({ field }) => (
                    <FormItem className="flex flex-row items-center justify-between rounded-lg border p-4">
                      <div className="space-y-0.5">
                        <FormLabel className="text-base">Enable fine-tuning</FormLabel>
                        <FormDescription>
                          Automatically splits questions for LoRA fine-tuning
                        </FormDescription>
                      </div>
                      <FormControl>
                        <Switch
                          checked={field.value}
                          onCheckedChange={field.onChange}
                        />
                      </FormControl>
                    </FormItem>
                  )}
                />
              </div>

              {form.watch("finetuneEnabled") && (
                <Card className="p-4 bg-muted/50">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <FormField
                      control={form.control}
                      name="finetunePercentage"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Test Set Percentage</FormLabel>
                          <Select onValueChange={(value) => field.onChange(parseFloat(value))} defaultValue={field.value.toString()}>
                            <FormControl>
                              <SelectTrigger>
                                <SelectValue />
                              </SelectTrigger>
                            </FormControl>
                            <SelectContent>
                              <SelectItem value="0.1">10% (Minimal)</SelectItem>
                              <SelectItem value="0.15">15%</SelectItem>
                              <SelectItem value="0.2">20% (Recommended)</SelectItem>
                              <SelectItem value="0.25">25%</SelectItem>
                              <SelectItem value="0.3">30% (Maximum)</SelectItem>
                            </SelectContent>
                          </Select>
                          <FormMessage />
                        </FormItem>
                      )}
                    />

                    <FormField
                      control={form.control}
                      name="finetuneSeed"
                      render={({ field }) => (
                        <FormItem>
                          <FormLabel>Random Seed</FormLabel>
                          <FormControl>
                            <Input
                              type="number"
                              min={0}
                              max={999999}
                              {...field}
                              onChange={(e) => {
                                const value = parseInt(e.target.value);
                                field.onChange(isNaN(value) ? undefined : value);
                              }}
                            />
                          </FormControl>
                          <FormDescription>
                            Random seed for reproducible train/test splits
                          </FormDescription>
                          <FormMessage />
                        </FormItem>
                      )}
                    />
                  </div>
                </Card>
              )}

              <FormField
                control={form.control}
                name="runName"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Run Name (Optional)</FormLabel>
                    <FormControl>
                      <Input placeholder="My evaluation run" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="space-y-4">
            {/* Primary Evaluation Button */}
            <div>
              <Button
                type="submit"
                disabled={isEvaluating}
                className="flex items-center gap-2 w-full sm:w-auto"
                size="lg"
              >
                {isEvaluating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    Start Full Evaluation
                  </>
                )}
              </Button>
            </div>

            {/* Local Testing Option */}
            <div className="border rounded-lg p-4 bg-gradient-to-r from-orange-50 to-red-50 border-orange-200">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <Zap className="h-4 w-4 text-orange-600" />
                    <span className="font-medium text-gray-900">Quick Local Test</span>
                    <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded-full">
                      No API Key Required
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">
                    Test with local Qwen model (up to 20 questions). Perfect for trying the system without external API costs.
                  </p>
                </div>
              </div>
              <Button
                type="button"
                variant="outline"
                onClick={startQwenEvaluation}
                disabled={isEvaluating}
                className="flex items-center gap-2 bg-white hover:bg-orange-50 border-orange-300 text-orange-700 hover:text-orange-800"
              >
                <Zap className="h-4 w-4" />
                Start Local Test
              </Button>
            </div>
          </div>
        </form>
      </Form>

      {/* Progress Section */}
      {isEvaluating && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={progress} className="w-full" />
            <div className="flex items-center gap-2">
              <div className="animate-pulse h-2 w-2 bg-blue-500 rounded-full"></div>
              <span className="text-sm text-muted-foreground">{statusMessage}</span>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Results Section */}
      {results && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Evaluation Results</CardTitle>
              <CardDescription>
                Run ID: <code className="text-xs bg-muted px-1 py-0.5 rounded">{currentRunId}</code>
              </CardDescription>
            </div>
            <Button onClick={downloadResults} variant="outline">
              Download Results
            </Button>
          </CardHeader>
          <CardContent>
            {results.aggregate_metrics ? (
              <div className="space-y-6">
                {/* Key Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-blue-600">
                      {(results.aggregate_metrics.mean_score || 0).toFixed(3)}
                    </div>
                    <div className="text-sm text-muted-foreground">Mean Score</div>
                  </div>
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <div className="text-2xl font-bold text-green-600">
                      {results.aggregate_metrics.num_samples || 0}
                    </div>
                    <div className="text-sm text-muted-foreground">Questions</div>
                  </div>
                  <div className="text-center p-4 bg-muted/50 rounded-lg">
                    <Badge variant={results.aggregate_metrics.statistically_significant ? "default" : "secondary"}>
                      {results.aggregate_metrics.statistically_significant ? "✓ Significant" : "✗ Not Significant"}
                    </Badge>
                    <div className="text-sm text-muted-foreground mt-1">
                      p={(results.aggregate_metrics.statistical_significance || 1).toFixed(3)}
                    </div>
                  </div>
                </div>

                {/* Additional Metrics */}
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="font-medium">95% Confidence Interval:</span>
                    <span>
                      [{results.aggregate_metrics.confidence_interval_95 ? 
                        results.aggregate_metrics.confidence_interval_95.map((v: number) => v.toFixed(3)).join(', ') : 
                        '0, 0'}]
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="font-medium">Score Range:</span>
                    <span>
                      {(results.aggregate_metrics.min_score || 0).toFixed(3)} - {(results.aggregate_metrics.max_score || 0).toFixed(3)}
                    </span>
                  </div>
                </div>

                {/* Mock Response Warning */}
                {results.individual_results && results.individual_results.some((r: any) => r.prediction && r.prediction.includes('Mock LLM response')) && (
                  <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <h4 className="font-medium text-yellow-800 mb-2">⚠️ Using Mock Responses</h4>
                    <p className="text-yellow-700 text-sm">
                      The system is using mock responses instead of real LLM evaluation.
                      To get real LLM responses, go to the Settings tab and configure your OpenRouter API key.
                    </p>
                  </div>
                )}

                {/* Individual Results Preview */}
                {results.individual_results && results.individual_results.length > 0 && (
                  <div>
                    <h4 className="font-medium mb-3">All Questions & Responses</h4>
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {results.individual_results.map((result: any, index: number) => (
                        <div key={index} className="p-4 border rounded-lg bg-muted/30">
                          <div className="font-medium mb-2">Question {index + 1}:</div>
                          <div className="text-sm mb-3">{result.question}</div>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                            <div>
                              <span className="font-medium text-green-600">Expected:</span>
                              <div className="mt-1">{result.ground_truth}</div>
                            </div>
                            <div>
                              <span className="font-medium text-blue-600">LLM Response:</span>
                              <div className="mt-1">{result.prediction}</div>
                            </div>
                          </div>
                          <div className="flex justify-end mt-3">
                            <Badge 
                              variant={result.score >= 0.8 ? "default" : result.score >= 0.6 ? "secondary" : "destructive"}
                            >
                              Score: {result.score.toFixed(2)} | Method: {result.method}
                            </Badge>
                          </div>
                        </div>
                      ))}
                    </div>
                    <div className="text-center mt-3 text-sm text-muted-foreground">
                      Showing all {results.individual_results.length} questions
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-center text-muted-foreground">
                Invalid results format received
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* Fine-tune Button - Appears after evaluation completion with fine-tuning enabled */}
      {evaluationCompleted && hasFineTuneData && currentRunId && (
        <Card className="border-2 border-gradient-to-r from-blue-500 to-purple-500 bg-gradient-to-r from-blue-50 to-purple-50">
          <CardContent className="p-6">
            <div className="text-center space-y-4">
              <div className="space-y-2">
                <h3 className="text-xl font-semibold text-blue-800">Ready for Fine-tuning!</h3>
                <p className="text-muted-foreground">
                  Your corpus has been evaluated and fine-tuning data has been prepared. 
                  You can now proceed with LoRA fine-tuning.
                </p>
              </div>
              <Button
                onClick={() => openFinetuningDashboard(currentRunId)}
                size="lg"
                className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-purple-500 hover:from-blue-600 hover:to-purple-600 text-white px-8 py-3"
              >
                <Settings className="h-5 w-5" />
                Start Fine-tuning
              </Button>
              <div className="text-xs text-muted-foreground">
                This will open the LoRA fine-tuning dashboard in a new tab
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}