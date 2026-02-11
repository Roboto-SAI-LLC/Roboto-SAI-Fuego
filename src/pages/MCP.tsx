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
    id: string;
    name: string;
    description: string;
    tools: Array<{ name: string; description: string }>;
    enabled: boolean;
}

interface MCPConfig {
    servers: MCPServer[];
}

const MCPPage = () => {
    const queryClient = useQueryClient();
    const [config, setConfig] = useState<MCPConfig | null>(null);

    const osAgentUrl = import.meta.env.VITE_OS_AGENT_URL || (typeof window !== 'undefined' && window.location.hostname === 'localhost' ? 'http://localhost:5055' : '');

    const { data, isLoading } = useQuery({
        queryKey: ["mcp-servers"],
        queryFn: async () => {
            if (!osAgentUrl) return { servers: [] } as MCPConfig;
            const res = await fetch(`${osAgentUrl}/api/servers`);
            if (!res.ok) throw new Error("Failed to fetch MCP servers");
            const json = await res.json();
            // OS agent wraps response in { success, data: { servers } }
            return { servers: json.data?.servers || json.servers || [] } as MCPConfig;
        },
        retry: false,
    });

    const updateConfig = useMutation({
        mutationFn: async ({ serverId, enabled }: { serverId: string; enabled: boolean }) => {
            const res = await fetch(`${osAgentUrl}/api/servers/toggle`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ serverId, enabled })
            });
            if (!res.ok) throw new Error("Failed to update server");
            return await res.json();
        },
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["mcp-servers"] });
            toast.success("Server status updated");
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

    const toggleServer = (serverId: string, enabled: boolean) => {
        updateConfig.mutate({ serverId, enabled });
    };

    if (isLoading) return <div className="p-10 text-center">Loading MCP Config...</div>;

    const servers = data?.servers || [];

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
                    {servers.map((server) => (
                        <motion.div
                            key={server.id}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            layout
                        >
                            <Card className={`border-l-4 ${!server.enabled ? 'border-l-muted' : 'border-l-green-500'} bg-card/50 backdrop-blur-sm`}>
                                <CardHeader className="flex flex-row items-center justify-between pb-2">
                                    <div className="space-y-1">
                                        <CardTitle className="text-lg flex items-center gap-2">
                                            <Server className="w-4 h-4" />
                                            {server.name}
                                        </CardTitle>
                                        <p className="text-sm text-muted-foreground">{server.description}</p>
                                    </div>
                                    <Switch
                                        checked={server.enabled}
                                        onCheckedChange={(enabled) => toggleServer(server.id, enabled)}
                                    />
                                </CardHeader>
                                <CardContent>
                                    <div className="text-sm text-muted-foreground mb-2">
                                        {server.tools.length} tools available
                                    </div>
                                    <div className="text-xs space-y-1">
                                        {server.tools.slice(0, 3).map((tool, idx) => (
                                            <div key={idx} className="truncate" title={tool.description}>
                                                • {tool.name}
                                            </div>
                                        ))}
                                        {server.tools.length > 3 && (
                                            <div className="text-muted-foreground">
                                                +{server.tools.length - 3} more...
                                            </div>
                                        )}
                                    </div>
                                    <div className="flex justify-between items-center mt-4">
                                        <span className={`text-xs px-2 py-1 rounded-full ${!server.enabled ? 'bg-muted text-muted-foreground' : 'bg-green-500/10 text-green-500'}`}>
                                            {server.enabled ? 'Active' : 'Disabled'}
                                        </span>
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
