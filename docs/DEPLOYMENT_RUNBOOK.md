# Deployment Runbook for Roboto SAI Platform

**Version:** 1.0  
**Date:** February 11, 2026  
**Owner:** DevOps/SRE Team

## Overview

This runbook provides step-by-step instructions for deploying Roboto SAI platform updates, including backend, frontend, and infrastructure changes.

## Prerequisites

### Access Requirements
- AWS/Cloud access with deployment permissions
- Supabase admin access
- Docker registry access
- CI/CD pipeline access

### Pre-deployment Checklist
- [ ] Code review completed and approved
- [ ] Tests passing (unit, integration, e2e)
- [ ] Security scan passed
- [ ] Database migrations tested in staging
- [ ] Rollback plan documented
- [ ] Communication plan prepared
- [ ] Maintenance window scheduled (if required)

## Deployment Environments

### Staging Environment
- **Purpose:** Pre-production testing
- **URL:** staging.roboto-sai.com
- **Database:** Supabase staging project
- **Monitoring:** Full observability stack

### Production Environment
- **Purpose:** Live user traffic
- **URL:** app.roboto-sai.com
- **Database:** Supabase production project
- **Monitoring:** Full observability with alerts

## Deployment Types

### Type 1: Frontend Only (Low Risk)
- Static asset updates
- UI/UX changes
- No backend changes

### Type 2: Backend API (Medium Risk)
- API endpoint changes
- Business logic updates
- Database schema changes

### Type 3: Infrastructure (High Risk)
- Database migrations
- New service deployments
- Configuration changes

## Step-by-Step Deployment

### Phase 1: Pre-deployment

#### 1.1 Environment Preparation
```bash
# Create deployment branch
git checkout -b deploy/$(date +%Y%m%d-%H%M%S)

# Update version numbers
echo "v$(date +%Y.%m.%d).$(git rev-parse --short HEAD)" > VERSION

# Build artifacts
docker build -t roboto-sai:latest .
docker tag roboto-sai:latest registry.roboto-sai.com/roboto-sai:$(git rev-parse --short HEAD)
```

#### 1.2 Database Backup
```bash
# Production database backup
supabase db dump --project-ref prod-project-id > backup-$(date +%Y%m%d).sql

# Verify backup integrity
pg_restore --list backup-$(date +%Y%m%d).sql > /dev/null
```

#### 1.3 Staging Deployment
```bash
# Deploy to staging
kubectl set image deployment/roboto-sai-backend backend=registry.roboto-sai.com/roboto-sai:$(git rev-parse --short HEAD) --namespace=staging

# Wait for rollout
kubectl rollout status deployment/roboto-sai-backend --namespace=staging --timeout=300s

# Run smoke tests
npm run test:staging
```

### Phase 2: Production Deployment

#### 2.1 Blue-Green Deployment Strategy
```bash
# Scale up blue environment
kubectl scale deployment roboto-sai-backend-blue --replicas=3 --namespace=production

# Wait for readiness
kubectl wait --for=condition=available --timeout=300s deployment/roboto-sai-backend-blue --namespace=production

# Switch traffic (Istio or similar)
kubectl apply -f production/traffic-blue.yaml

# Monitor traffic switch
kubectl get virtualservice roboto-sai -o yaml
```

#### 2.2 Database Migration
```bash
# Apply migrations (if any)
supabase db push --project-ref prod-project-id

# Verify migration success
supabase db diff --project-ref prod-project-id --schema public
```

#### 2.3 Post-deployment Verification
```bash
# Health checks
curl -f https://app.roboto-sai.com/api/health

# API smoke tests
npm run test:production

# Performance monitoring
# Check response times, error rates, resource usage
```

### Phase 3: Post-deployment

#### 3.1 Monitoring and Alerts
- Monitor error rates (< 1%)
- Check response times (< 500ms p95)
- Verify database connections
- Review application logs

#### 3.2 Rollback Procedures
If issues detected:

```bash
# Immediate rollback
kubectl apply -f production/traffic-green.yaml

# Scale down problematic deployment
kubectl scale deployment roboto-sai-backend-blue --replicas=0 --namespace=production

# Restore database if needed
supabase db restore --project-ref prod-project-id backup-$(date +%Y%m%d).sql
```

#### 3.3 Communication
- Update status page
- Notify stakeholders
- Document any issues encountered

## Monitoring and Alerting

### Key Metrics to Monitor
- **Availability:** Service uptime (target: 99.9%)
- **Performance:** API response times (target: <500ms p95)
- **Errors:** Error rate (target: <1%)
- **Resources:** CPU/memory usage (target: <80%)
- **Database:** Connection pool usage, query performance

### Alert Conditions
- Error rate > 5% for 5 minutes
- Response time > 2s for 10 minutes
- Service unavailable for 1 minute
- Database connections > 90%

## Troubleshooting

### Common Issues

#### Issue: Pod CrashLoopBackOff
```bash
# Check pod logs
kubectl logs -f deployment/roboto-sai-backend --namespace=production

# Check resource limits
kubectl describe pod <pod-name> --namespace=production

# Check configuration
kubectl get configmap roboto-sai-config -o yaml
```

#### Issue: Database Connection Issues
```bash
# Check database connectivity
kubectl exec -it <pod-name> -- nc -zv db-host 5432

# Verify credentials
kubectl get secret db-secret -o yaml

# Check connection pool
# Monitor Supabase dashboard
```

#### Issue: Slow API Responses
```bash
# Check resource utilization
kubectl top pods --namespace=production

# Review application logs
kubectl logs -f deployment/roboto-sai-backend --namespace=production --since=1h

# Profile performance
# Use APM tools or add profiling code
```

## Rollback Runbook

### Automatic Rollback
If deployment health checks fail:

1. Alert triggered
2. Traffic automatically switched to previous version
3. Investigation begins
4. Manual rollback confirmed or fix deployed

### Manual Rollback
```bash
# Scale down current deployment
kubectl scale deployment roboto-sai-backend-blue --replicas=0 --namespace=production

# Scale up previous deployment
kubectl scale deployment roboto-sai-backend-green --replicas=3 --namespace=production

# Switch traffic back
kubectl apply -f production/traffic-green.yaml

# Verify rollback success
curl -f https://app.roboto-sai.com/api/health
```

## Maintenance Procedures

### Regular Maintenance
- **Weekly:** Log rotation, temporary file cleanup
- **Monthly:** Security updates, dependency updates
- **Quarterly:** Full backup testing, disaster recovery testing

### Emergency Maintenance
- Schedule during low-traffic windows
- Communicate maintenance windows 48h in advance
- Have rollback plan ready
- Monitor systems during maintenance

## Contact Information

- **Primary On-call:** devops@roboto-sai.com
- **Secondary:** sre@roboto-sai.com
- **Escalation:** cto@roboto-sai.com
- **24/7 Support:** +1 (555) 123-4567

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-11 | Initial runbook creation |