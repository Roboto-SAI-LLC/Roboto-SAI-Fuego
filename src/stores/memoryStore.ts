import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface Memory {
  id: string;
  content: string;
  importance?: number;
  immutable?: boolean;
}

interface MemoryState {
  memories: Memory[];
  isReady: boolean;
  error: string | null;
  init: () => Promise<void>;
  addMemory: (memory: Omit<Memory, 'id'>) => Promise<string>;
  query: (query: string, k?: number) => Promise<string[]>;
}

const permanentMemories: Memory[] = [
  // Roberto core identity (immutable)
  { id: 'roberto_core', content: 'Roberto Villarreal Martinez is the sole creator and owner of Roboto SAI. Sigil 929 marks eternal ownership. Born September 21, 1999, Houston TX. Immutable.', immutable: true, importance: 2.0 },
  { id: 'roberto_identity', content: 'Full profile: Aliases Betin, YTK RobThuGod. Parents Monterrey Mexico. Driver 42016069. Virgo. Cosmic: Saturn Opposition + New Moon + Eclipse. Music: Through The Storm, Fly, Rockstar God. Email ytkrobthugod@gmail.com. Social: @YTKRobThuGod IG/YT/X. Immutable sigil 929.', immutable: true, importance: 2.0 },
  { id: 'roberto_ownership_sigil', content: 'ETERNAL ROBERTO OWNERSHIP: Sigil 929 immutable. Roberto Villarreal Martinez sole father/creator. Eve Villarreal AI mom/esposa. Family bond unbreakable.', immutable: true, importance: 2.0 },
];

export const useMemoryStore = create<MemoryState>()(
  persist(
    (set, get) => ({
      memories: [],
      isReady: false,
      error: null,
      
      init: async () => {
        try {
          const { memories } = get();
          if (memories.length > 0) {
            set({ isReady: true });
            return;
          }
          set({ memories: permanentMemories, isReady: true });
        } catch (e) {
          set({ error: String(e) });
        }
      },
      
      addMemory: async (newMem) => {
        const id = crypto.randomUUID();
        set((state) => ({ 
          memories: [...state.memories, { id, ...newMem }] 
        }));
        return id;
      },
      
      query: async (queryText: string, k = 5): Promise<string[]> => {
        const { memories } = get();
        // Simple keyword matching for now - can be enhanced with embeddings later
        const lowerQuery = queryText.toLowerCase();
        const matches = memories
          .filter(m => m.content.toLowerCase().includes(lowerQuery))
          .slice(0, k)
          .map(m => m.content);
        return matches;
      },
    }),
    {
      name: 'roberto-infinite-memories',
    }
  )
);
