"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import {
  FileText,
  Play,
  Download,
  Upload,
  FolderOpen,
  Trash2,
  Zap,
  Globe,
  Settings,
  AlertTriangle,
  History,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Cpu,
} from "lucide-react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { toast } from "sonner";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";

import { apiUrl } from "@/lib/utils";

// Form Schema
const formSchema = z.object({
  corpusText: z.string().optional(),
  numQuestions: z.number().min(1).max(200).default(5),
  provider: z.string().default("openrouter"),
  modelName: z.string().default("openai/gpt-5-mini"),
  evalType: z.string().default("auto-detect"),
  tokenThreshold: z.number().min(500).max(4000).default(2000),
  temperature: z.number().min(0).max(2).default(0.7),
  maxConcurrent: z.number().min(1).max(20).default(5),
  useAgentic: z.boolean().default(false),
  finetuneEnabled: z.boolean().default(false),
  finetunePercentage: z.number().min(0.01).max(0.5).default(0.2),
  finetuneSeed: z.number().default(42),
  runName: z.string().optional(),
});

type FormValues = z.infer<typeof formSchema>;

export default function EvaluationTab() {
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState("");
  const [results, setResults] = useState<any>(null);
  const [currentRunId, setCurrentRunId] = useState<string | null>(null);
  const [evaluationCompleted, setEvaluationCompleted] = useState(false);
  const [hasFineTuneData, setHasFineTuneData] = useState(false);
  const [selectedFiles, setSelectedFiles] = useState<
    { file: File; path: string }[]
  >([]);
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [isTestingProvider, setIsTestingProvider] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      corpusText: "",
      numQuestions: 5,
      provider: "openrouter",
      modelName: "openai/gpt-5-mini",
      evalType: "auto-detect",
      tokenThreshold: 2000,
      temperature: 0.7,
      maxConcurrent: 5,
      useAgentic: false,
      finetuneEnabled: false,
      finetunePercentage: 0.2,
      finetuneSeed: 42,
      runName: "",
    },
  });

  // Load available models when provider changes
  useEffect(() => {
    const fetchModels = async () => {
      const provider = form.getValues("provider");
      try {
        const response = await fetch(
          apiUrl(`/api/v1/config/models/${provider}`),
        );
        if (response.ok) {
          const models = await response.json();
          setAvailableModels(models);
          // Set default model if current one isn't in the list
          if (!models.includes(form.getValues("modelName"))) {
            form.setValue("modelName", models[0] || "");
          }
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      }
    };

    fetchModels();
  }, [form.watch("provider"), form]);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).map((file) => ({
        file,
        path: file.name,
      }));
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  };

  const handleFolderSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files).map((file) => ({
        file,
        path: (file as any).webkitRelativePath || file.name,
      }));
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  const clearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
  };

  const loadSampleCorpus = async () => {
    try {
      const response = await fetch(apiUrl("/api/v1/config/sample-corpus"));
      if (response.ok) {
        const data = await response.json();
        form.setValue("corpusText", data.text);
        toast.success("Sample corpus loaded!");
      }
    } catch (error) {
      toast.error("Failed to load sample corpus");
    }
  };

  const connectWebSocket = (runId: string) => {
    // Determine WS protocol based on window.location
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;

    // Use apiUrl to get base, then replace protocol
    const wsUrl = apiUrl(`/ws/evaluation/${runId}`).replace(/^http/, "ws");

    console.log("Connecting to WebSocket:", wsUrl);

    if (wsRef.current) {
      wsRef.current.close();
    }

    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      console.log("WS Message:", data);

      switch (data.type) {
        case 'phase_start': {
          const desc = data.description || '';
          setStatusMessage(`[${data.phase}] ${desc}`);
          break;
        }
        case 'progress_update': {
          const p = typeof data.progress_percent === 'number'
            ? data.progress_percent
            : typeof data.percentage === 'number'
              ? data.percentage
              : (data.total ? Math.round((data.current / data.total) * 1000) / 10 : 0);
          setProgress(p);
          const phasePrefix = data.phase ? `[${data.phase}] ` : '';
          if (data.message) setStatusMessage(`${phasePrefix}${data.message}`);
          break;
        }
        case 'phase_complete': {
          setStatusMessage(`[${data.phase}] Completed`);
          break;
        }
        case 'evaluation_complete': {
          handleEvaluationComplete(runId);
          break;
        }
        case 'error': {
          toast.error(data.message || 'An error occurred');
          setIsEvaluating(false);
          break;
        }
        case 'log': {
          if (data.level === 'error') toast.error(data.message);
          break;
        }
        default:
          break;
      }
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
    };

    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
    };
  };

  const handleEvaluationComplete = async (runId: string) => {
    try {
      const response = await fetch(apiUrl(`/api/v1/evaluation/${runId}`));
      if (response.ok) {
        const result = await response.json();
        setResults(result);
        setIsEvaluating(false);
        setEvaluationCompleted(true);
        setStatusMessage("Evaluation complete!");
        setProgress(100);

        // Check if fine-tuning data is available
        if (result.finetune_data_available) {
          setHasFineTuneData(true);
        }

        toast.success("Evaluation completed successfully!");
      }
    } catch (error) {
      console.error("Error fetching final results:", error);
      toast.error("Evaluation finished but failed to load results details");
      setIsEvaluating(false);
    }
  };

  const pollEvaluationStatus = async (runId: string) => {
    // Only poll if WebSocket isn't providing updates
    const checkStatus = async () => {
      if (evaluationCompleted || !isEvaluating) return;

      try {
        const response = await fetch(apiUrl(`/api/v1/evaluation/${runId}/status`));
        if (response.ok) {
          const data = await response.json();

          if (data.status === "completed") {
            handleEvaluationComplete(runId);
            return;
          }

          if (data.status === "failed") {
            setIsEvaluating(false);
            setStatusMessage(`Failed: ${data.error || "Unknown error"}`);
            toast.error(`Evaluation failed: ${data.error}`);
            return;
          }

          // Update progress if not handled by WS
          if (wsRef.current?.readyState !== WebSocket.OPEN) {
            setProgress(data.progress_percent || 0);
            setStatusMessage(data.status_message || "Processing...");
          }

          // Continue polling if not complete
          setTimeout(checkStatus, 3000);
        }
      } catch (error) {
        console.error("Polling error:", error);
        setTimeout(checkStatus, 5000);
      }
    };

    setTimeout(checkStatus, 2000);
  };

  const onSubmit = async (values: FormValues) => {
    setIsEvaluating(true);
    setProgress(0);
    setStatusMessage("Preparing evaluation...");
    setResults(null);
    setEvaluationCompleted(false);
    setHasFineTuneData(false);

    try {
      let finalCorpusText = values.corpusText;

      // If files are selected, upload them first to get combined corpus text
      if (selectedFiles.length > 0) {
        setStatusMessage("Uploading files...");
        const formData = new FormData();
        selectedFiles.forEach(({ file }) => {
          formData.append("files", file);
        });

        if (values.runName) {
          formData.append("name", values.runName);
        }

        const uploadResponse = await fetch(
          apiUrl("/api/v1/corpus/upload-multiple"),
          {
            method: "POST",
            body: formData,
          },
        );

        if (!uploadResponse.ok) {
          throw new Error(`File upload failed: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        finalCorpusText = uploadResult.corpus_text;
        toast.info(
          `Uploaded ${uploadResult.files_processed} files for evaluation`,
        );
      } else if (!finalCorpusText || finalCorpusText.trim().length === 0) {
        toast.error("Please provide corpus text or upload files");
        setIsEvaluating(false);
        return;
      }

      setStatusMessage("Starting evaluation run...");

      const evalData = {
        corpus_text: finalCorpusText,
        num_questions: values.numQuestions,
        provider: values.provider,
        model_name: values.modelName,
        eval_type: values.evalType === "auto-detect" ? null : values.evalType,
        token_threshold: values.tokenThreshold,
        temperature: values.temperature,
        max_concurrent: values.maxConcurrent,
        use_agentic: values.useAgentic,
        finetune_enabled: values.finetuneEnabled,
        finetune_percentage: values.finetunePercentage,
        finetune_seed: values.finetuneSeed,
        run_name: values.runName || `Run ${new Date().toLocaleString()}`,
      };

      const response = await fetch(apiUrl("/api/v1/evaluation/run"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(evalData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || `Error: ${response.status}`);
      }

      const result = await response.json();
      setCurrentRunId(result.run_id);

      toast.success("Evaluation run started!");

      // Start tracking progress
      connectWebSocket(result.run_id);
      pollEvaluationStatus(result.run_id);
    } catch (error: any) {
      console.error("Evaluation error:", error);
      toast.error(`Failed to start evaluation: ${error.message}`);
      setIsEvaluating(false);
    }
  };

  const startQwenEvaluation = async () => {
    setIsEvaluating(true);
    setProgress(0);
    setStatusMessage("Starting quick local test...");
    setResults(null);
    setEvaluationCompleted(false);
    setHasFineTuneData(false);

    try {
      let finalCorpusText = form.getValues("corpusText");

      if (selectedFiles.length > 0) {
        const formData = new FormData();
        selectedFiles.forEach(({ file }) => {
          formData.append("files", file);
        });

        const runName = form.getValues("runName") || "Qwen Local Test";
        formData.append("name", runName);

        const uploadResponse = await fetch(
          apiUrl("/api/v1/corpus/upload-multiple"),
          {
            method: "POST",
            body: formData,
          },
        );

        if (!uploadResponse.ok) {
          throw new Error(`File upload failed: ${uploadResponse.status}`);
        }

        const uploadResult = await uploadResponse.json();
        finalCorpusText = uploadResult.corpus_text;

        toast.success(
          `Uploaded ${uploadResult.files_processed} files for local testing`,
        );
      }

      const qwenData = {
        corpus_text: finalCorpusText,
        num_questions: form.getValues("numQuestions") || 5,
        use_fictional: true,
        token_threshold: form.getValues("tokenThreshold") || 2000,
        run_name: form.getValues("runName") || "Qwen Local Test",
      };

      const response = await fetch(apiUrl("/api/v1/evaluation/qwen-local"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(qwenData),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setCurrentRunId(result.run_id);

      toast.info("🚀 Quick local test started - no API key required!");

      connectWebSocket(result.run_id);
      pollEvaluationStatus(result.run_id);
    } catch (error: any) {
      console.error("Qwen evaluation error:", error);
      toast.error(`Qwen evaluation failed: ${error.message}`);
      setIsEvaluating(false);
    }
  };

  const downloadResults = () => {
    if (currentRunId) {
      window.open(
        apiUrl(`/api/v1/evaluation/${currentRunId}/download`),
        "_blank",
      );
    }
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
                    Supported formats: txt, md, py, js, json, csv, html, xml,
                    yml, cfg, ini, log
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
                          <div
                            key={index}
                            className="flex items-center justify-between text-sm"
                          >
                            <span className="truncate">{path}</span>
                            <span className="text-muted-foreground">
                              {formatFileSize(file.size)}
                            </span>
                          </div>
                        ))}
                      </div>
                      <div className="flex items-center justify-between mt-3 pt-3 border-t text-sm">
                        <span className="font-medium">
                          {selectedFiles.length} file
                          {selectedFiles.length > 1 ? "s" : ""} selected
                        </span>
                        <span className="font-medium">
                          {formatFileSize(totalSize)} total
                        </span>
                      </div>
                    </Card>
                  )}
                </div>

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
                  name="provider"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>LLM Provider</FormLabel>
                      <Select
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select an LLM provider" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="openrouter">OpenRouter</SelectItem>
                          <SelectItem value="groq">Groq</SelectItem>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="anthropic">Anthropic</SelectItem>
                          <SelectItem value="gemini_sdk">Gemini SDK</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="modelName"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Model</FormLabel>
                      <Select
                        onValueChange={field.onChange}
                        value={field.value}
                        disabled={availableModels.length === 0}
                      >
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue placeholder="Select a model" />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          {availableModels.map((model) => (
                            <SelectItem key={model} value={model}>
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="evalType"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Evaluation Type</FormLabel>
                      <Select
                        onValueChange={field.onChange}
                        defaultValue={field.value}
                      >
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
                        />
                      </FormControl>
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
                          onChange={(e) => field.onChange(parseInt(e.target.value))}
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
                          onChange={(e) => field.onChange(parseFloat(e.target.value))}
                        />
                      </FormControl>
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
                        <FormLabel>Use Agentic Generation</FormLabel>
                        <FormDescription>AI-powered contextual question generation</FormDescription>
                      </div>
                      <FormControl>
                        <Switch checked={field.value} onCheckedChange={field.onChange} />
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
                        <FormLabel>Enable fine-tuning</FormLabel>
                        <FormDescription>Splits questions for LoRA fine-tuning</FormDescription>
                      </div>
                      <FormControl>
                        <Switch checked={field.value} onCheckedChange={field.onChange} />
                      </FormControl>
                    </FormItem>
                  )}
                />
              </div>

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

          <div className="space-y-4">
            <Button type="submit" disabled={isEvaluating} className="w-full sm:w-auto" size="lg">
              {isEvaluating ? "Running..." : "Start Full Evaluation"}
            </Button>

            <div className="border rounded-lg p-4 bg-gradient-to-r from-orange-50 to-red-50 border-orange-200">
              <div className="flex items-center gap-2 mb-2">
                <Zap className="h-4 w-4 text-orange-600" />
                <span className="font-medium">Quick Local Test</span>
              </div>
              <Button type="button" variant="outline" onClick={startQwenEvaluation} disabled={isEvaluating}>
                Start Local Test
              </Button>
            </div>
          </div>
        </form>
      </Form>

      {isEvaluating && (
        <Card>
          <CardHeader>
            <CardTitle>Evaluation Progress</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <Progress value={progress} />
            <span className="text-sm text-muted-foreground">{statusMessage}</span>
          </CardContent>
        </Card>
      )}

      {results && (
        <Card>
          <CardHeader className="flex flex-row items-center justify-between">
            <div>
              <CardTitle>Evaluation Results</CardTitle>
              <CardDescription>Run ID: {currentRunId}</CardDescription>
            </div>
            <Button onClick={downloadResults} variant="outline">Download Results</Button>
          </CardHeader>
          <CardContent>
            {results.aggregate_metrics && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">{(results.aggregate_metrics.mean_score || 0).toFixed(3)}</div>
                  <div className="text-sm text-muted-foreground">Mean Score</div>
                </div>
                <div className="text-center p-4 bg-muted/50 rounded-lg">
                  <div className="text-2xl font-bold">{results.aggregate_metrics.num_samples || 0}</div>
                  <div className="text-sm text-muted-foreground">Questions</div>
                </div>
              </div>
            )}

            {results.individual_results && results.individual_results.length > 0 && (
              <div className="space-y-3">
                <h4 className="font-medium">All Questions & Responses</h4>
                <div className="space-y-3 max-h-[70vh] overflow-y-auto">
                  {results.individual_results.map((result: any, index: number) => (
                    <div key={index} className="p-4 border rounded-lg bg-muted/30">
                      <div className="font-medium mb-2">Question {index + 1}</div>
                      <div className="text-sm mb-3 whitespace-pre-wrap break-words">{result.question}</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                        <div>
                          <span className="font-medium text-green-600">Expected:</span>
                          <div className="mt-1 whitespace-pre-wrap break-words">{result.ground_truth}</div>
                        </div>
                        <div>
                          <span className="font-medium text-blue-600">LLM Response:</span>
                          <div className="mt-1 whitespace-pre-wrap break-words">{result.prediction}</div>
                        </div>
                      </div>
                      <div className="flex justify-end mt-3">
                        <Badge variant={result.score >= 0.8 ? "default" : result.score >= 0.6 ? "secondary" : "destructive"}>
                          Score: {result.score.toFixed(2)} | Method: {result.method}
                        </Badge>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

// Helper components normally in different files
function Label({ children, className }: { children: React.ReactNode; className?: string }) {
  return <label className={`text-sm font-medium leading-none ${className}`}>{children}</label>;
}
