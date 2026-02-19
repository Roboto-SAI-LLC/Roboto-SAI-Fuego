import { useEffect, useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { ArrowLeft, Mail, Flame } from 'lucide-react';
import { toast } from 'sonner';

import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { EmberParticles } from '@/components/effects/EmberParticles';
import { AuthForm } from '@/components/auth/AuthForm';
import { useAuthStore } from '@/stores/authStore';

const Register = () => {
  const navigate = useNavigate();
  const { register, refreshSession, username } = useAuthStore();
  const isLoggedIn = useAuthStore((state) => state.isLoggedIn);
  const [pendingEmail, setPendingEmail] = useState<string | null>(null);

  useEffect(() => {
    if (isLoggedIn) {
      navigate('/chat', { replace: true });
    }
  }, [isLoggedIn, navigate]);

  const handleAuthSubmit = async (data: { username: string; email: string; password: string }) => {
    try {
      const result = await register(data.email, data.password);
      if (result.pendingVerification) {
        // Email confirmation required — show check-inbox screen
        setPendingEmail(data.email);
        return;
      }
      // Confirmation disabled in Supabase — session should be active immediately
      await refreshSession();
      toast.success('Welcome to the eternal flame!');
      navigate('/chat');
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Registration failed';
      toast.error(message);
    }
  };

  // Check-your-inbox screen
  if (pendingEmail) {
    return (
      <div className="min-h-screen bg-background relative overflow-hidden flex items-center justify-center px-4">
        <EmberParticles count={20} />
        <div className="absolute top-6 left-6 z-10">
          <Link to="/">
            <Button variant="ghost" className="gap-2">
              <ArrowLeft className="w-4 h-4" />
              Back
            </Button>
          </Link>
        </div>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="relative z-10 w-full flex justify-center"
        >
          <Card className="w-full max-w-md bg-card/80 backdrop-blur-md border-fire/30 shadow-2xl">
            <CardHeader className="text-center pb-2">
              <motion.div
                animate={{ scale: [1, 1.05, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
                className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br from-primary/20 to-fire/20 border border-fire/30 mx-auto mb-3"
              >
                <Flame className="w-6 h-6 text-fire" />
              </motion.div>
              <CardTitle className="text-2xl font-display text-fire">Check Your Inbox</CardTitle>
              <CardDescription className="text-muted-foreground">
                Account created — one step left
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="flex flex-col items-center gap-3 py-4 text-center">
                <div className="flex items-center justify-center w-14 h-14 rounded-full bg-fire/10 border border-fire/20">
                  <Mail className="w-7 h-7 text-fire" />
                </div>
                <p className="text-sm text-foreground/80 leading-relaxed">
                  We sent a verification link to{' '}
                  <span className="font-semibold text-fire">{pendingEmail}</span>.
                  <br />
                  Click the link in that email to activate your account, then sign in.
                </p>
                <p className="text-xs text-muted-foreground">
                  Didn't get it? Check your spam folder or try signing up again.
                </p>
              </div>
              <Button
                className="w-full"
                variant="outline"
                onClick={() => navigate('/login')}
              >
                Go to Sign In
              </Button>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background relative overflow-hidden flex items-center justify-center px-4">
      <EmberParticles count={20} />

      <div className="absolute top-6 left-6 z-10">
        <Link to="/">
          <Button variant="ghost" className="gap-2">
            <ArrowLeft className="w-4 h-4" />
            Back
          </Button>
        </Link>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="relative z-10 w-full flex justify-center"
      >
        <AuthForm onSubmit={handleAuthSubmit} defaultUsername={username ?? ''} initialMode="register" />
      </motion.div>
    </div>
  );
};

export default Register;
