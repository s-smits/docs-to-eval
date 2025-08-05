"use client";

import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Settings, Key, TestTube, Save, RefreshCw, Trash2, Eye, EyeOff, ExternalLink, Info } from "lucide-react";
import { toast } from "sonner";

const settingsSchema = z.object({
  provider: z.string().min(1, "Provider is required"),
  modelName: z.string().min(1, "Model name is required"),
  apiKey: z.string().optional(),
  maxTokens: z.number().min(1).max(131072).default(32768),
  defaultTemperature: z.number().min(0).max(2).default(0.7),
});

type SettingsFormData = z.infer<typeof settingsSchema>;

interface Config {
  llm: {
    provider: string;
    model_name: string;
    api_key?: string;
    max_tokens: number;
    temperature: number;
  };
}

interface TestResult {
  status: string;
  message: string;
}

export function SettingsTab() {
  const [currentConfig, setCurrentConfig] = useState<Config | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isTesting, setIsTesting] = useState(false);
  const [testResult, setTestResult] = useState<TestResult | null>(null);
  const [showApiKey, setShowApiKey] = useState(false);
  const [apiKeyStatus, setApiKeyStatus] = useState<{ configured: boolean; message: string }>({
    configured: false,
    message: "No API key configured"
  });

  const form = useForm<SettingsFormData>({
    resolver: zodResolver(settingsSchema),
    defaultValues: {
      provider: "openrouter",
      modelName: "anthropic/claude-sonnet-4",
      maxTokens: 32768,
      defaultTemperature: 0.7,
    },
  });

  const providerHelp: Record<string, { text: string; url: string; credits?: string }> = {
    openrouter: {
      text: "Get your OpenRouter API key from openrouter.ai/keys",
      url: "https://openrouter.ai/keys",
      credits: "https://openrouter.ai/credits"
    },
    openai: {
      text: "Get your OpenAI API key from platform.openai.com/api-keys",
      url: "https://platform.openai.com/api-keys"
    },
    anthropic: {
      text: "Get your Anthropic API key from console.anthropic.com",
      url: "https://console.anthropic.com/"
    }
  };

  const loadCurrentConfig = async () => {
    setIsLoading(true);
    try {
      const response = await fetch('/api/v1/config/current');
      
      if (!response.ok) {
        throw new Error(`Failed to load config: ${response.status}`);
      }
      
      const config = await response.json();
      setCurrentConfig(config);

      // Update form fields
      form.setValue("provider", config.llm.provider || "openrouter");
      form.setValue("modelName", config.llm.model_name || "anthropic/claude-sonnet-4");
      form.setValue("maxTokens", config.llm.max_tokens || 32768);
      form.setValue("defaultTemperature", config.llm.temperature || 0.7);

      // Load API key from sessionStorage if available
      const storedApiKey = sessionStorage.getItem('docs_to_eval_api_key');
      if (storedApiKey && !form.getValues("apiKey")) {
        form.setValue("apiKey", storedApiKey);
      }

      // Update API key status
      const hasApiKey = config.llm.api_key && config.llm.api_key !== '***masked***';
      const hasStoredKey = storedApiKey || form.getValues("apiKey");

      if (hasApiKey || hasStoredKey) {
        setApiKeyStatus({
          configured: true,
          message: "API key configured ✓"
        });
      } else {
        setApiKeyStatus({
          configured: false,
          message: "No API key configured"
        });
      }

    } catch (error: any) {
      console.error('Failed to load config:', error);
      toast.error('Failed to load configuration');
    } finally {
      setIsLoading(false);
    }
  };

  const testApiKey = async () => {
    const apiKey = form.getValues("apiKey");
    const provider = form.getValues("provider");
    const model = form.getValues("modelName");

    if (!apiKey?.trim()) {
      toast.warning('Please enter an API key first');
      return;
    }

    setIsTesting(true);
    setTestResult(null);

    try {
      const response = await fetch('/api/v1/config/test-api-key', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          api_key: apiKey,
          provider: provider,
          model: model
        })
      });

      const result = await response.json();
      setTestResult(result);

      if (result.status === 'success') {
        toast.success('API key test successful!');
      } else {
        toast.error(`API key test failed: ${result.message}`);
      }

    } catch (error: any) {
      const errorResult = {
        status: 'error',
        message: `Test failed: ${error.message}`
      };
      setTestResult(errorResult);
      toast.error(`API key test failed: ${error.message}`);
    } finally {
      setIsTesting(false);
    }
  };

  const clearApiKey = () => {
    form.setValue("apiKey", "");
    sessionStorage.removeItem('docs_to_eval_api_key');
    setApiKeyStatus({
      configured: false,
      message: "No API key configured"
    });
    toast.info("API key cleared");
  };

  const onSubmit = async (data: SettingsFormData) => {
    try {
      const configUpdate = {
        llm: {
          provider: data.provider,
          model_name: data.modelName,
          max_tokens: data.maxTokens,
          temperature: data.defaultTemperature
        }
      };

      if (data.apiKey?.trim()) {
        (configUpdate.llm as any).api_key = data.apiKey;
        // Store API key in sessionStorage for persistence
        sessionStorage.setItem('docs_to_eval_api_key', data.apiKey);
      }

      const response = await fetch('/api/v1/config/update', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(configUpdate)
      });

      const result = await response.json();

      if (result.status === 'success') {
        toast.success('Settings saved successfully!');

        // Update API key status
        if (result.api_key_set) {
          setApiKeyStatus({
            configured: true,
            message: "API key configured ✓"
          });
        }

        // Clear the API key input for security
        form.setValue("apiKey", "");

        // Reload current config to reflect changes
        loadCurrentConfig();

      } else {
        toast.error(result.message || 'Failed to save settings');
      }

    } catch (error: any) {
      console.error('Settings save error:', error);
      toast.error('Failed to save settings: ' + error.message);
    }
  };

  const currentProvider = form.watch("provider");
  const providerInfo = providerHelp[currentProvider];

  useEffect(() => {
    loadCurrentConfig();
  }, []);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="flex items-center gap-3">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"></div>
          <span className="text-muted-foreground">Loading configuration...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* API Key Status */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Key className="h-5 w-5" />
            API Configuration Status
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-3">
            <div className={`h-3 w-3 rounded-full ${apiKeyStatus.configured ? 'bg-green-500' : 'bg-gray-400'}`}></div>
            <span className={apiKeyStatus.configured ? 'text-green-700' : 'text-muted-foreground'}>
              {apiKeyStatus.message}
            </span>
            {apiKeyStatus.configured && (
              <Badge variant="outline" className="text-green-700 border-green-200">
                Ready
              </Badge>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Settings Form */}
      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                LLM Provider Settings
              </CardTitle>
              <CardDescription>
                Configure your preferred language model provider and settings
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="provider"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>LLM Provider</FormLabel>
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="openrouter">OpenRouter (Recommended)</SelectItem>
                          <SelectItem value="openai">OpenAI</SelectItem>
                          <SelectItem value="anthropic">Anthropic</SelectItem>
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
                      <Select onValueChange={field.onChange} defaultValue={field.value}>
                        <FormControl>
                          <SelectTrigger>
                            <SelectValue />
                          </SelectTrigger>
                        </FormControl>
                        <SelectContent>
                          <SelectItem value="anthropic/claude-sonnet-4">Claude Sonnet 4 (Latest & Best)</SelectItem>
                          <SelectItem value="google/gemini-2.5-flash">Gemini 2.5 Flash (Fast)</SelectItem>
                          <SelectItem value="google/gemini-2.5-pro">Gemini 2.5 Pro (Balanced)</SelectItem>
                          <SelectItem value="openai/gpt-4.1">GPT-4.1 (Latest OpenAI)</SelectItem>
                        </SelectContent>
                      </Select>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>

              <FormField
                control={form.control}
                name="apiKey"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>API Key</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Input
                          {...field}
                          type={showApiKey ? "text" : "password"}
                          placeholder="Enter your API key here"
                          className="pr-10"
                        />
                        <Button
                          type="button"
                          variant="ghost"
                          size="sm"
                          className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                          onClick={() => setShowApiKey(!showApiKey)}
                        >
                          {showApiKey ? (
                            <EyeOff className="h-4 w-4" />
                          ) : (
                            <Eye className="h-4 w-4" />
                          )}
                        </Button>
                      </div>
                    </FormControl>
                    {providerInfo && (
                      <FormDescription className="flex flex-col gap-2">
                        <div>
                          {providerInfo.text}{" "}
                          <a
                            href={providerInfo.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800"
                          >
                            {providerInfo.url.replace('https://', '')}
                            <ExternalLink className="h-3 w-3" />
                          </a>
                        </div>
                        {providerInfo.credits && (
                          <div className="text-orange-600">
                            <strong>Note:</strong> You need credits in your account. Add credits at{" "}
                            <a
                              href={providerInfo.credits}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center gap-1 hover:text-orange-800"
                            >
                              {providerInfo.credits.replace('https://', '')}
                              <ExternalLink className="h-3 w-3" />
                            </a>
                          </div>
                        )}
                      </FormDescription>
                    )}
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <FormField
                  control={form.control}
                  name="maxTokens"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Max Output Tokens</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          min={1}
                          max={131072}
                          {...field}
                          onChange={(e) => field.onChange(parseInt(e.target.value))}
                        />
                      </FormControl>
                      <FormDescription>
                        Maximum number of tokens in the response (1-131072)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />

                <FormField
                  control={form.control}
                  name="defaultTemperature"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Default Temperature</FormLabel>
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
                      <FormDescription>
                        Controls randomness (0.0 = deterministic, 2.0 = very random)
                      </FormDescription>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </CardContent>
          </Card>

          {/* Action Buttons */}
          <div className="flex gap-3 flex-wrap">
            <Button
              type="button"
              variant="outline"
              onClick={testApiKey}
              disabled={isTesting}
              className="flex items-center gap-2"
            >
              {isTesting ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-current"></div>
                  Testing...
                </>
              ) : (
                <>
                  <TestTube className="h-4 w-4" />
                  Test API Key
                </>
              )}
            </Button>
            <Button type="submit" className="flex items-center gap-2">
              <Save className="h-4 w-4" />
              Save Settings
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={loadCurrentConfig}
              className="flex items-center gap-2"
            >
              <RefreshCw className="h-4 w-4" />
              Reload Config
            </Button>
            <Button
              type="button"
              variant="outline"
              onClick={clearApiKey}
              className="flex items-center gap-2 text-destructive hover:text-destructive"
            >
              <Trash2 className="h-4 w-4" />
              Clear API Key
            </Button>
          </div>
        </form>
      </Form>

      {/* Test Result */}
      {testResult && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TestTube className="h-5 w-5" />
              API Key Test Result
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className={`p-4 rounded-lg border ${
              testResult.status === 'success' 
                ? 'bg-green-50 border-green-200 text-green-800' 
                : 'bg-red-50 border-red-200 text-red-800'
            }`}>
              <div className="flex items-center gap-2">
                {testResult.status === 'success' ? (
                  <div className="h-2 w-2 bg-green-500 rounded-full"></div>
                ) : (
                  <div className="h-2 w-2 bg-red-500 rounded-full"></div>
                )}
                <span className="font-medium">{testResult.message}</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Environment Variables Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            Environment Variables
          </CardTitle>
          <CardDescription>
            Alternative configuration method using environment variables
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="bg-muted p-4 rounded-lg font-mono text-sm space-y-2">
            <div>export OPENROUTER_API_KEY="your-api-key-here"</div>
            <div>export DOCS_TO_EVAL_MODEL_NAME="anthropic/claude-sonnet-4"</div>
            <div>export DOCS_TO_EVAL_PROVIDER="openrouter"</div>
          </div>
          <p className="text-sm text-muted-foreground mt-3">
            Environment variables take precedence over web interface settings and are more secure for production deployments.
          </p>
        </CardContent>
      </Card>

      {/* Agentic Evaluation Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            About Agentic Evaluation
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <p className="text-sm text-muted-foreground">
            <strong>When "Use Agentic Generation" is enabled:</strong>
          </p>
          <ul className="text-sm text-muted-foreground space-y-1 list-disc list-inside ml-4">
            <li>Uses your configured LLM to generate contextual questions</li>
            <li>Creates domain-specific benchmarks from your corpus</li>
            <li>Produces higher quality, more relevant evaluation questions</li>
            <li>Requires a valid API key to function</li>
          </ul>
          <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
            <p className="text-sm text-blue-700">
              <strong>Note:</strong> Without an API key, the system will use mock/sample questions for demonstration purposes.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}