import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { Header } from "@/components/layout/Header";
import { Database, Server, RefreshCw, Power } from "lucide-react";
import { motion } from "framer-motion";

interface MCPServer {
    command?: string;
    args?: string[];
    type?: string;
    url?: string;
    disabled?: boolean;
}

interface MCPConfig {
    mcpServers: Record<string, MCPServer>;
}

const MCPPage = () => {
    const queryClient = useQueryClient();
    const [config, setConfig] = useState<MCPConfig | null>(null);

    const { data, isLoading } = useQuery({
        queryKey: ["mcp-config"],
        queryFn: async () => {
            const res = await fetch(`${import.meta.env.VITE_API_URL || ""}/api/mcp/config`);
            if (!res.ok) throw new Error("Failed to fetch config");
            return await res.json() as MCPConfig;
        }
    });

    const updateConfig = useMutation({
        mutationFn: async (newConfig: MCPConfig) => {
            const res = await fetch(`${import.meta.env.VITE_API_URL || ""}/api/mcp/config`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(newConfig)
            });
            if (!res.ok) throw new Error("Failed to update config");
            return await res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["mcp-config"] });
            toast.success("Configuration updated locally (Check backend logs for persistence)");
        },
        onError: (err) => {
            toast.error("Failed to update: " + err.message);
        }
    });

    const restartMCP = useMutation({
        mutationFn: async () => {
            const res = await fetch(`${import.meta.env.VITE_API_URL || ""}/api/mcp/restart`, { method: "POST" });
            if (!res.ok) throw new Error("Failed to restart");
            return await res.json();
        },
        onSuccess: () => {
            toast.success("MCP Services Restart Triggered");
        }
    });

    const toggleServer = (name: string, current: boolean) => {
        if (!data) return;
        const newConfig = { ...data };
        if (!newConfig.mcpServers[name]) return;

        newConfig.mcpServers[name] = {
            ...newConfig.mcpServers[name],
            disabled: !current
        };
        updateConfig.mutate(newConfig);
    };

    if (isLoading) return <div className="p-10 text-center">Loading MCP Config...</div>;

    const servers = data?.mcpServers || {};

    return (
        <div className="min-h-screen bg-background text-foreground">
            <Header />
            <div className="container mx-auto p-6 pt-24 max-w-5xl">
                <div className="flex justify-between items-center mb-8">
                    <div>
                        <h1 className="text-3xl font-display text-primary mb-2">MCP Tools Management</h1>
                        <p className="text-muted-foreground">Manage your Model Context Protocol servers and connections.</p>
                    </div>
                    <Button
                        variant="outline"
                        onClick={() => restartMCP.mutate()}
                        className="gap-2 border-primary/20 hover:bg-primary/10"
                    >
                        <RefreshCw className={`w-4 h-4 ${restartMCP.isPending ? 'animate-spin' : ''}`} />
                        Restart Services
                    </Button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {Object.entries(servers).map(([name, server]) => (
                        <motion.div
                            key={name}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            layout
                        >
                            <Card className={`border-l-4 ${server.disabled ? 'border-l-muted' : 'border-l-green-500'} bg-card/50 backdrop-blur-sm`}>
                                <CardHeader className="flex flex-row items-center justify-between pb-2">
                                    <div className="space-y-1">
                                        <CardTitle className="text-lg flex items-center gap-2">
                                            {server.type === 'http' ? <Server className="w-4 h-4" /> : <Database className="w-4 h-4" />}
                                            {name}
                                        </CardTitle>
                                    </div>
                                    <Switch
                                        checked={!server.disabled}
                                        onCheckedChange={() => toggleServer(name, !server.disabled)}
                                    />
                                </CardHeader>
                                <CardContent>
                                    <div className="text-sm font-mono bg-muted/50 p-2 rounded mb-2 overflow-x-auto">
                                        {server.url || (server.command ? `${server.command} ${server.args?.join(' ')}` : 'Unknown')}
                                    </div>
                                    <div className="flex justify-between items-center mt-4">
                                        <span className={`text-xs px-2 py-1 rounded-full ${server.disabled ? 'bg-muted text-muted-foreground' : 'bg-green-500/10 text-green-500'}`}>
                                            {server.disabled ? 'Disabled' : 'Active'}
                                        </span>
                                        {/* Edit button could go here */}
                                    </div>
                                </CardContent>
                            </Card>
                        </motion.div>
                    ))}

                    {/* Add Server Card Placeholder */}
                    <Card className="border-dashed border-2 flex items-center justify-center p-6 h-full min-h-[200px] hover:border-primary/50 cursor-pointer transition-colors group">
                        <div className="text-center">
                            <div className="w-12 h-12 rounded-full bg-muted group-hover:bg-primary/10 flex items-center justify-center mx-auto mb-4 transition-colors">
                                <span className="text-2xl text-muted-foreground group-hover:text-primary">+</span>
                            </div>
                            <p className="text-muted-foreground font-medium group-hover:text-primary">Add New Integration</p>
                        </div>
                    </Card>
                </div>
            </div>
        </div>
    );
};

export default MCPPage;
