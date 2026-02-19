/**
 * Roboto SAI Auth Store
 * Simple client-side auth for user-specific chat persistence
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { supabase } from '../lib/supabase';
import { config } from '../config';

type AuthUser = {
  id: string;
  email: string;
  display_name?: string | null;
  avatar_url?: string | null;
  provider?: string | null;
};

type MeResponse = {
  user?: AuthUser;
};

type PersistedAuthState = {
  userId: string | null;
  username: string | null;
  email: string | null;
  avatarUrl: string | null;
  provider: string | null;
  isLoggedIn: boolean;
};

const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null;

const coercePersistedAuthState = (value: unknown): PersistedAuthState => {
  if (!isRecord(value)) {
    return {
      userId: null,
      username: null,
      email: null,
      avatarUrl: null,
      provider: null,
      isLoggedIn: false,
    };
  }

  return {
    userId: typeof value.userId === 'string' ? value.userId : null,
    username: typeof value.username === 'string' ? value.username : null,
    email: typeof value.email === 'string' ? value.email : null,
    avatarUrl: typeof value.avatarUrl === 'string' ? value.avatarUrl : null,
    provider: typeof value.provider === 'string' ? value.provider : null,
    isLoggedIn: value.isLoggedIn === true,
  };
};

interface AuthState {
  userId: string | null;
  username: string | null;
  email: string | null;
  avatarUrl: string | null;
  provider: string | null;
  isLoggedIn: boolean;

  loginWithPassword: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<{ pendingVerification: boolean }>;

  // Real backend session auth
  refreshSession: () => Promise<boolean>;
  requestMagicLink: (email: string) => Promise<void>;
  logout: () => Promise<void>;
  updateUsername: (newUsername: string) => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist<AuthState, [], [], PersistedAuthState>(
    (set) => ({
      userId: null,
      username: null,
      email: null,
      avatarUrl: null,
      provider: null,
      isLoggedIn: false,

      refreshSession: async () => {
        // Use backend auth endpoint to check session
        try {
          const meUrl = `${config.apiBaseUrl}/api/auth/me`;
          const response = await fetch(meUrl, {
            method: 'GET',
            credentials: 'include',
          });

          if (response.ok) {
            const data = await response.json();
            if (data.user) {
              set({
                userId: data.user.id,
                username: data.user.display_name || data.user.email?.split('@')[0] || null,
                email: data.user.email,
                avatarUrl: data.user.avatar_url || null,
                provider: data.user.provider || 'supabase',
                isLoggedIn: true,
              });
              return true;
            }
          }
        } catch (error) {
          console.warn('Session refresh failed:', error);
        }

        // Session invalid or request failed
        set({ userId: null, username: null, email: null, avatarUrl: null, provider: null, isLoggedIn: false });
        return false;
      },

      register: async (email: string, password: string): Promise<{ pendingVerification: boolean }> => {
        // Use backend auth endpoint to get session cookie
        const registerUrl = `${config.apiBaseUrl}/api/auth/register`;

        const response = await fetch(registerUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Registration failed' }));
          throw new Error(errorData.detail || 'Registration failed');
        }

        const data = await response.json().catch(() => ({})) as Record<string, unknown>;
        // 202 = email confirmation required; 200/201 = immediately active (email confirm disabled)
        const pendingVerification = response.status === 202 || data.pending_verification === true;
        return { pendingVerification };
      },

      loginWithPassword: async (email: string, password: string) => {
        // Use backend auth endpoint to get session cookie
        const loginUrl = `${config.apiBaseUrl}/api/auth/login`;

        const response = await fetch(loginUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ email, password }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Login failed' }));
          throw new Error(errorData.detail || 'Login failed');
        }

        const data = await response.json();
        if (data.success && data.user) {
          set({
            userId: data.user.id,
            username: data.user.display_name || data.user.email?.split('@')[0] || null,
            email: data.user.email,
            avatarUrl: data.user.avatar_url || null,
            provider: data.user.provider || 'supabase',
            isLoggedIn: true,
          });
        } else {
          throw new Error('Login response missing user data');
        }
      },

      requestMagicLink: async (email: string) => {
        // Use backend auth endpoint for magic links
        const magicUrl = `${config.apiBaseUrl}/api/auth/magic/request`;

        const response = await fetch(magicUrl, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ email }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Magic link request failed' }));
          throw new Error(errorData.detail || 'Magic link request failed');
        }
      },

      logout: async () => {
        // Use backend auth endpoint to clear session cookie
        try {
          const logoutUrl = `${config.apiBaseUrl}/api/auth/logout`;
          await fetch(logoutUrl, {
            method: 'POST',
            credentials: 'include',
          });
        } catch (error) {
          console.warn('Backend logout failed, clearing local state anyway:', error);
        }

        // Clear local state regardless of backend response
        set({ userId: null, username: null, email: null, avatarUrl: null, provider: null, isLoggedIn: false });
      },

      updateUsername: async (newUsername: string) => {
        // Validate username on frontend
        if (!newUsername || newUsername.trim().length === 0) {
          throw new Error('Username cannot be empty');
        }

        const trimmedUsername = newUsername.trim();

        if (trimmedUsername.length < 2) {
          throw new Error('Username must be at least 2 characters long');
        }

        if (trimmedUsername.length > 20) {
          throw new Error('Username cannot exceed 20 characters');
        }

        if (!/^[a-zA-Z0-9_-\s]+$/.test(trimmedUsername)) {
          throw new Error('Display name can only contain letters, numbers, underscores, hyphens, and spaces');
        }

        // Use backend auth endpoint to update username
        const updateUrl = `${config.apiBaseUrl}/api/auth/update`;

        const response = await fetch(updateUrl, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'include',
          body: JSON.stringify({ display_name: trimmedUsername }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({ detail: 'Update failed' }));
          throw new Error(errorData.detail || 'Update failed');
        }

        // Update local state
        set({ username: trimmedUsername });
      },
    }),
    {
      name: 'robo-auth',
      version: 2,
      migrate: (persistedState, _version) => {
        const envelope = isRecord(persistedState) ? persistedState : null;
        const rawState = envelope && isRecord(envelope.state) ? envelope.state : persistedState;

        const parsed = coercePersistedAuthState(rawState);
        if (parsed.provider === 'demo') {
          return {
            userId: null,
            username: null,
            email: null,
            avatarUrl: null,
            provider: null,
            isLoggedIn: false,
          };
        }
        return parsed;
      },
    }
  )
);
