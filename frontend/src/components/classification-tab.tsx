"use client";

import { useState, useRef } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import * as z from "zod";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Form, FormControl, FormDescription, FormField, FormItem, FormLabel, FormMessage } from "@/components/ui/form";
import { Upload, FolderOpen, Trash2, Search, Sparkles } from "lucide-react";
import { toast } from "sonner";

const classificationSchema = z.object({
  corpusText: z.string().min(10, "Corpus text must be at least 10 characters"),
});

type ClassificationFormData = z.infer<typeof classificationSchema>;

interface FileWithPath {
  file: File;
  path: string;
}

interface ClassificationResult {
  primary_type: string;
  confidence: number;
  secondary_types: string[];
  analysis: string;
  reasoning: string;
}

export function ClassificationTab() {
  const [selectedFiles, setSelectedFiles] = useState<FileWithPath[]>([]);
  const [isClassifying, setIsClassifying] = useState(false);
  const [results, setResults] = useState<ClassificationResult | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const form = useForm<ClassificationFormData>({
    resolver: zodResolver(classificationSchema),
    defaultValues: {
      corpusText: "",
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

    toast.success(`Selected ${files.length} file${files.length > 1 ? 's' : ''} for classification`);
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

    toast.success(`Selected ${files.length} file${files.length > 1 ? 's' : ''} from folder for classification`);
  };

  const clearFiles = () => {
    setSelectedFiles([]);
    if (fileInputRef.current) fileInputRef.current.value = "";
    if (folderInputRef.current) folderInputRef.current.value = "";
    toast.info("Files cleared");
  };

  const demoClassification = () => {
    // Demo the notification system
    toast.info('📡 This is an info notification with auto-removal after 5 seconds!');
    setTimeout(() => toast.success('🎉 Success! Your operation completed successfully!'), 1000);
    setTimeout(() => toast.warning('⚠️ Warning: This is a longer warning message'), 2000);
    setTimeout(() => toast.error('❌ Error notifications stay visible longer for important messages'), 3000);
  };

  const onSubmit = async (data: ClassificationFormData) => {
    let corpusText = data.corpusText;

    if (!corpusText.trim() && selectedFiles.length === 0) {
      toast.error('Please enter corpus text or select files/folder to classify');
      return;
    }

    setIsClassifying(true);
    setResults(null);

    try {
      let result;

      if (selectedFiles.length > 0) {
        // Upload multiple files for classification
        const formData = new FormData();
        selectedFiles.forEach(({ file }) => {
          formData.append('files', file);
        });
        formData.append('name', 'Classification corpus');

        const response = await fetch('/api/v1/corpus/upload-multiple', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`File upload failed: ${response.status}`);
        }

        result = await response.json();
        toast.success(`Uploaded and classified ${selectedFiles.length} files`);
      } else {
        // Use text input for classification
        const response = await fetch('/api/v1/corpus/upload', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            text: corpusText,
            name: 'Classification corpus'
          })
        });

        if (!response.ok) {
          throw new Error(`Classification failed: ${response.status}`);
        }

        result = await response.json();
        toast.success('Corpus classified successfully');
      }

      setResults(result);

    } catch (error: any) {
      console.error('Classification error:', error);
      toast.error('Classification failed: ' + error.message);
    } finally {
      setIsClassifying(false);
    }
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 0.8) return "bg-green-500";
    if (confidence >= 0.6) return "bg-yellow-500";
    return "bg-red-500";
  };

  const getConfidenceLabel = (confidence: number) => {
    if (confidence >= 0.8) return "High";
    if (confidence >= 0.6) return "Medium";
    return "Low";
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
                <Search className="h-5 w-5" />
                Corpus Classification
              </CardTitle>
              <CardDescription>
                Analyze and classify your corpus to determine the best evaluation type
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <FormField
                control={form.control}
                name="corpusText"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Corpus Text (Optional)</FormLabel>
                    <FormControl>
                      <Textarea
                        {...field}
                        placeholder="Enter corpus text to classify or use file upload below..."
                        className="min-h-[120px]"
                      />
                    </FormControl>
                    <FormDescription>
                      Provide text content to analyze and classify automatically
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />

              <div className="space-y-3">
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
                    onClick={demoClassification}
                    className="flex items-center gap-2"
                  >
                    <Sparkles className="h-4 w-4" />
                    Demo Notifications
                  </Button>
                </div>
                <p className="text-sm text-muted-foreground">
                  Supported formats: txt, md, py, js, json, csv, html, xml, yml, cfg, ini, log
                </p>
              </div>

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
            </CardContent>
          </Card>

          {/* Action Button */}
          <Button
            type="submit"
            disabled={isClassifying}
            className="flex items-center gap-2"
          >
            {isClassifying ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                Classifying...
              </>
            ) : (
              <>
                <Search className="h-4 w-4" />
                Classify Corpus
              </>
            )}
          </Button>
        </form>
      </Form>

      {/* Results Section */}
      {results && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Search className="h-5 w-5" />
              Classification Results
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Primary Classification */}
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold">Primary Classification</h3>
                <div className="flex items-center gap-2">
                  <div className={`h-3 w-3 rounded-full ${getConfidenceColor(results.confidence)}`}></div>
                  <Badge variant="outline">
                    {getConfidenceLabel(results.confidence)} Confidence ({(results.confidence * 100).toFixed(1)}%)
                  </Badge>
                </div>
              </div>
              
              <div className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg border">
                <div className="text-2xl font-bold text-blue-600 mb-2">
                  {results.primary_type.split('_').map(word => 
                    word.charAt(0).toUpperCase() + word.slice(1)
                  ).join(' ')}
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`h-2 rounded-full ${getConfidenceColor(results.confidence)}`}
                    style={{ width: `${results.confidence * 100}%` }}
                  ></div>
                </div>
              </div>
            </div>

            {/* Secondary Types */}
            {results.secondary_types && results.secondary_types.length > 0 && (
              <div className="space-y-3">
                <h3 className="text-lg font-semibold">Secondary Classifications</h3>
                <div className="flex flex-wrap gap-2">
                  {results.secondary_types.map((type, index) => (
                    <Badge key={index} variant="secondary">
                      {type.split('_').map(word => 
                        word.charAt(0).toUpperCase() + word.slice(1)
                      ).join(' ')}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Analysis */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Analysis</h3>
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm leading-relaxed">{results.analysis}</p>
              </div>
            </div>

            {/* Reasoning */}
            <div className="space-y-3">
              <h3 className="text-lg font-semibold">Reasoning</h3>
              <div className="p-4 bg-muted/50 rounded-lg">
                <p className="text-sm leading-relaxed">{results.reasoning}</p>
              </div>
            </div>

            {/* Recommendations */}
            <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
              <h4 className="font-medium text-blue-800 mb-2">💡 Recommendations</h4>
              <div className="text-blue-700 text-sm space-y-1">
                <p>• Use "<strong>{results.primary_type}</strong>" as the evaluation type for best results</p>
                <p>• Consider the secondary types if the primary doesn't yield good results</p>
                <p>• Confidence level indicates how certain the classification is</p>
                {results.confidence < 0.6 && (
                  <p className="text-yellow-700">⚠️ <strong>Low confidence:</strong> Consider providing more specific corpus content for better classification</p>
                )}
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}