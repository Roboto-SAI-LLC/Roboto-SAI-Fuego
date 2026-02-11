# Monitoring Runbook for Roboto SAI Platform

**Version:** 1.0  
**Date:** February 11, 2026  
**Owner:** SRE Team

## Overview

This runbook outlines monitoring procedures for the Roboto SAI platform, including metrics collection, alerting, and incident response.

## Monitoring Architecture

### Components
- **Application Metrics:** Custom metrics from FastAPI backend
- **Infrastructure Metrics:** Kubernetes, Docker container metrics
- **Database Metrics:** Supabase performance monitoring
- **External API Metrics:** xAI Grok API monitoring
- **User Experience:** Frontend performance monitoring

### Tools
- **Prometheus:** Metrics collection and storage
- **Grafana:** Visualization and dashboards
- **Alertmanager:** Alert routing and management
- **ELK Stack:** Log aggregation and analysis

## Key Metrics

### Application Metrics

#### API Performance
```python
# Request duration histogram
REQUEST_DURATION = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint', 'status_code']
)

# Request rate
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total number of HTTP requests',
    ['method', 'endpoint', 'status_code']
)
```

#### Business Metrics
- Active users (DAU/MAU)
- Chat sessions created
- Messages processed
- AI inference requests

#### Error Metrics
- 4xx/5xx error rates
- Database connection errors
- External API failures

### Infrastructure Metrics

#### Container Metrics
- CPU usage percentage
- Memory usage/RSS
- Network I/O
- Disk I/O

#### Kubernetes Metrics
- Pod status and restarts
- Resource requests/limits
- Node capacity and utilization

### Database Metrics

#### Supabase Metrics
- Query execution time
- Connection pool utilization
- Database size and growth
- Replication lag

#### Query Performance
```sql
-- Slow query monitoring
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

## Alerting Rules

### Critical Alerts (Page immediately)

#### Service Down
```
alert: ServiceDown
expr: up{job="roboto-sai-backend"} == 0
for: 1m
labels:
  severity: critical
annotations:
  summary: "Roboto SAI backend is down"
  description: "Backend service has been down for 1 minute"
```

#### High Error Rate
```
alert: HighErrorRate
expr: rate(http_requests_total{status_code=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.05
for: 5m
labels:
  severity: critical
annotations:
  summary: "High 5xx error rate: {{ $value | printf "%.2f" }}"
  description: "Error rate above 5% for 5 minutes"
```

#### Database Connection Issues
```
alert: DatabaseConnectionError
expr: rate(database_connection_errors_total[5m]) > 5
for: 2m
labels:
  severity: critical
annotations:
  summary: "Database connection errors"
  description: "More than 5 connection errors in 5 minutes"
```

### Warning Alerts (Investigate within 30 minutes)

#### High Latency
```
alert: HighLatency
expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
for: 5m
labels:
  severity: warning
annotations:
  summary: "High API latency: {{ $value | printf "%.2f" }}s"
  description: "95th percentile response time above 2 seconds"
```

#### Resource Utilization
```
alert: HighCPUUsage
expr: rate(container_cpu_usage_seconds_total[5m]) / rate(container_spec_cpu_quota[5m]) > 0.8
for: 10m
labels:
  severity: warning
annotations:
  summary: "High CPU usage on {{ $labels.pod }}"
  description: "CPU usage above 80% for 10 minutes"
```

## Incident Response

### Alert Triage

#### Step 1: Acknowledge Alert
1. Check alert dashboard in Grafana
2. Acknowledge in Alertmanager
3. Assess severity and impact

#### Step 2: Gather Information
```bash
# Check recent logs
kubectl logs -f deployment/roboto-sai-backend --namespace=production --since=10m

# Check pod status
kubectl get pods --namespace=production

# Check database connections
# Supabase dashboard or metrics
```

#### Step 3: Diagnose Issue
Common scenarios:

**High Error Rate:**
- Check application logs for error patterns
- Verify external API (xAI) status
- Check database connectivity

**High Latency:**
- Review performance metrics
- Check resource utilization
- Profile slow endpoints

**Service Down:**
- Check pod status and events
- Verify deployment status
- Check node health

#### Step 4: Mitigate
Depending on issue:

**For API Issues:**
```bash
# Scale up deployment
kubectl scale deployment roboto-sai-backend --replicas=5 --namespace=production

# Restart problematic pods
kubectl delete pod <pod-name> --namespace=production
```

**For Database Issues:**
- Check Supabase status page
- Contact Supabase support if needed
- Consider failover procedures

#### Step 5: Resolution
- Apply fix or rollback if necessary
- Monitor for 30 minutes post-resolution
- Update incident tracking system

#### Step 6: Post-mortem
- Document root cause
- Identify preventive measures
- Update monitoring/alerts if needed

## Dashboards

### Production Dashboard
- Service health overview
- Key metrics (latency, error rate, throughput)
- Resource utilization
- Database performance

### Detailed API Dashboard
- Per-endpoint metrics
- Error breakdown by status code
- Geographic distribution
- User agent analysis

### Business Dashboard
- User activity metrics
- Chat session analytics
- AI usage statistics
- Revenue metrics (if applicable)

## Log Management

### Log Aggregation
- All application logs sent to ELK stack
- Structured logging with consistent format
- Log retention: 90 days

### Log Analysis
```bash
# Search for errors
kubectl logs -l app=roboto-sai-backend --namespace=production | grep ERROR

# Analyze request patterns
# Use Kibana for log analysis
```

### Alerting on Logs
- Error rate anomalies
- Security event detection
- Performance issue patterns

## Synthetic Monitoring

### Uptime Checks
- External monitoring service pings /api/health every 30 seconds
- Multi-region monitoring (US East, EU West, Asia Pacific)

### API Tests
- Automated tests run every 5 minutes:
  - Health check endpoint
  - Basic chat API test
  - Database connectivity test

### Performance Tests
- Weekly synthetic load tests
- Measure API performance under load
- Validate SLO compliance

## SLO/SLA Monitoring

### Service Level Objectives
- **Availability:** 99.9% uptime
- **Latency:** P95 < 500ms for API calls
- **Error Rate:** < 1% of requests

### SLA Commitments
- 99.5% uptime for paid customers
- Response time < 1 second P95
- Support response within 4 hours

## On-call Rotation

### Primary On-call
- Monday-Friday: 9 AM - 5 PM PST
- Weekends/Holidays: Best effort

### Escalation Path
1. Primary on-call (15 minutes)
2. Secondary SRE (30 minutes)
3. Engineering Manager (1 hour)
4. CTO (2 hours)

### Handover Process
- Daily standup for status updates
- Clear documentation of ongoing issues
- Knowledge transfer sessions

## Maintenance Windows

### Scheduled Maintenance
- Monthly security updates: Second Tuesday, 2-4 AM PST
- Weekly dependency updates: Sunday 1-2 AM PST
- Emergency maintenance: As needed with 24h notice

### Maintenance Procedures
1. Schedule window with stakeholders
2. Set maintenance page
3. Perform maintenance
4. Test thoroughly
5. Remove maintenance page

## Contact Information

- **SRE Team:** sre@roboto-sai.com
- **On-call Pager:** +1 (555) 987-6543
- **Status Page:** status.roboto-sai.com
- **Chat Channel:** #incidents (Slack)

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-11 | Initial monitoring runbook |